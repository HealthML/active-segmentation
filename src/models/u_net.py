"""U-Net architecture."""

from collections import OrderedDict
from typing import List

import torch
from torch import nn
import torch.nn.functional as F


class UNet(nn.Module):
    """
    This U-Net implementation was originally taken from `this implementation
    <https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py>`_
    and adapted to a flexible number of levels and for optional 3d mode.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels. Should be equal to the number of classes (for
            multi-label segmentation tasks excluding the background class).
        multi_label (bool, optional): Whether the model should produce single-label or multi-label outputs. If set to
            `False`, the model's predictions are computed using a Softmax activation layer. to If set to `True`, sigmoid
            activation layers are used to compute the model's predicitions. Defaults to False.
        init_features (int, optional): Number of feature channels of the first U-Net block,
            in each down-sampling block, the number of feature channels is doubled. Defaults to 32.
        num_levels (int, optional): Number levels (encoder and decoder blocks) in the U-Net. Defaults to 4.
        dim (int, optional): The dimensionality of the U-Net. Defaults to 2.
        p_dropout (float, optional): Probability of applying dropout to the outputs of the encoder layers. Defaults to
            0.
    """

    # pylint: disable-msg=too-many-instance-attributes

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        multi_label: bool = False,
        init_features: int = 32,
        num_levels: int = 4,
        dim: int = 2,
        p_dropout: float = 0,
    ):

        super().__init__()

        self.num_levels = num_levels

        MaxPool = nn.MaxPool2d if dim == 2 else nn.MaxPool3d
        ConvTranspose = nn.ConvTranspose2d if dim == 2 else nn.ConvTranspose3d
        Conv = nn.Conv2d if dim == 2 else nn.Conv3d

        features = init_features

        self.encoders = nn.ModuleList(
            [
                UNet._block(
                    in_channels if level == 0 else features * (2 ** (level - 1)),
                    features * (2 ** level),
                    name=f"enc{level + 1}",
                    dim=dim,
                    p_dropout=p_dropout,
                )
                for level in range(num_levels)
            ]
        )
        self.pools = nn.ModuleList(
            [MaxPool(kernel_size=2, stride=2) for _ in range(num_levels)]
        )

        self.bottleneck = UNet._block(
            features * (2 ** (num_levels - 1)),
            features * (2 ** num_levels),
            name="bottleneck",
            dim=dim,
        )

        self.upconvs = nn.ModuleList(
            [
                ConvTranspose(
                    features * (2 ** (level + 1)),
                    features * (2 ** level),
                    kernel_size=2,
                    stride=2,
                )
                for level in range(num_levels)
            ]
        )
        self.decoders = nn.ModuleList(
            [
                UNet._block(
                    features * (2 ** (level + 1)),
                    features * (2 ** level),
                    name=f"dec{level + 1}",
                    dim=dim,
                )
                for level in range(num_levels)
            ]
        )

        self.conv = Conv(in_channels=features, out_channels=out_channels, kernel_size=1)

        self.prediction_layer = (
            torch.nn.Sigmoid() if multi_label else torch.nn.Softmax(dim=-1 * (dim + 1))
        )

    def forward(self, x):
        """

        Args:
            x (Tensor): Batch of input images.

        Returns:
            Tensor: Segmentation masks.
        """

        x = x.float()
        # individually store encoding results for skip connections
        encodings: List[torch.Tensor] = []
        for level in range(self.num_levels):
            encodings.append(
                self.encoders[level](
                    x if level == 0 else self.pools[level - 1](encodings[level - 1])
                )
            )

        bottleneck = self.bottleneck(self.pools[-1](encodings[-1]))

        dec = bottleneck
        for level in reversed(range(self.num_levels)):
            dec = self.upconvs[level](dec)

            # for the relevant dimensions [0, 0] for even and [0, 1] for odd in reversed order flattened
            pad = [
                p for dim in reversed(encodings[level].size()[2:]) for p in [0, dim % 2]
            ]
            dec = F.pad(dec, tuple(pad), "constant", 0)

            dec = torch.cat((dec, encodings[level]), dim=1)
            dec = self.decoders[level](dec)

        decoded = self.conv(dec)

        return self.prediction_layer(decoded)

    @staticmethod
    def _block(
        in_channels: int, features: int, name: str, dim: int, p_dropout: float = 0
    ):

        Conv = nn.Conv2d if dim == 2 else nn.Conv3d
        BatchNorm = nn.BatchNorm2d if dim == 2 else nn.BatchNorm3d

        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        Conv(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", BatchNorm(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        Conv(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", BatchNorm(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                    (name + "dropout", nn.Dropout(p=p_dropout)),
                ]
            )
        )
