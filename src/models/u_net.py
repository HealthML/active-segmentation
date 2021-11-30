"""U-Net architecture."""

from collections import OrderedDict
from typing import Tuple

import torch
from torch import nn
import numpy as np


class UNet(nn.Module):
    """
    This U-Net implementation was originally taken from
    https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py
    and adapted to a flexible number of levels and for optinal 3d mode.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (should be equal to the number of classes
            excluding the background)
        init_features: Number of feature channels of the first U-Net block,
            in each down-sampling block, the number of feature channels is doubled.
        num_levels: Number levels (encoder and decoder blocks) in the U-Net.
        input_shape: The input shape of the U-Net.
    """

    # pylint: disable-msg=too-many-instance-attributes

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        init_features: int = 32,
        num_levels: int = 4,
        input_shape: Tuple[int] = (240, 240),
    ):

        super().__init__()

        self.num_levels = num_levels

        dim = len(input_shape)

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
                    output_padding=UNet.upconv_output_padding(level, input_shape),
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
        for level in reversed(range(self.num_levels)):
            dec = self.upconvs[level](dec)
            dec = torch.cat((dec, encs[level]), dim=1)
            dec = self.decoders[level](dec)

        return torch.sigmoid(self.conv(dec))

    @staticmethod
    def _block(in_channels: int, features: int, name: str, dim: int):

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
                ]
            )
        )

    @staticmethod
    def upconv_output_padding(level: int, input_shape: Tuple[int]) -> Tuple[int]:
        """
        Calculates the output padding for transpose convolutions to match the output size to the corresponding encoding
        step for concatination.

        Args:
            level (int): The level in the UNet the transpose convolution is on.
            input_shape (Tuple[int]): The input shape for the whole UNet.

        Returns:
            Tuple[int]: The output padding for the transpose convolution.
        """
        shape = np.asarray(input_shape)
        for _ in range(level):
            shape = np.floor_divide(shape, 2)
        odd = shape % 2
        return tuple(odd)
