"""U-Net architecture."""

from collections import OrderedDict

import torch
from torch import nn


class UNet(nn.Module):
    """
    This U-Net implementation was originally taken from
    https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py
    and adapted to a flexible number of levels.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (should be equal to the number of classes excluding the background)
        init_features: Number of feature channels of the first U-Net block, in each down-sampling block, the number of
            feature channels is doubled.
        num_levels: Number levels (encoder and decoder blocks) in the U-Net.
    """

    # pylint: disable-msg=too-many-instance-attributes

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        init_features: int = 32,
        num_levels: int = 4,
    ):

        super().__init__()

        self.num_levels = num_levels

        features = init_features

        self.encoders = nn.ModuleList(
            [
                UNet._block(
                    in_channels if level == 0 else features * (2 ** (level - 1)),
                    features * (2 ** level),
                    name=f"enc{level + 1}",
                )
                for level in range(num_levels)
            ]
        )
        self.pools = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=2, stride=2) for _ in range(num_levels)]
        )

        self.bottleneck = UNet._block(
            features * (2 ** (num_levels - 1)),
            features * (2 ** num_levels),
            name="bottleneck",
        )

        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose2d(
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
                )
                for level in range(num_levels)
            ]
        )

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        """

        Args:
            x (Tensor): Batch of input images.

        Returns:
            Tensor: Segmentation masks.
        """

        x = x.float()
        encs = []  # individually store encoding results for skip connections
        for level in range(self.num_levels):
            encs.append(
                self.encoders[level](
                    x if level == 0 else self.pools[level - 1](encs[level - 1])
                )
            )

        bottleneck = self.bottleneck(self.pools[-1](encs[-1]))

        dec = bottleneck
        for i in reversed(range(self.num_levels)):
            dec = self.upconvs[i](dec)
            dec = torch.cat((dec, encs[i]), dim=1)
            dec = self.decoders[i](dec)

        return torch.sigmoid(self.conv(dec))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
