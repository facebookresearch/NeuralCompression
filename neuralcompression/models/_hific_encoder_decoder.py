# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from neuralcompression.layers import ChannelNorm2D


def _channel_norm_2d(input_channels, affine=True):
    return ChannelNorm2D(
        input_channels,
        affine=affine,
    )


class HiFiCEncoder(torch.nn.Module):
    """
    High-Fidelity Generative Image Compression (HiFiC) encoder.

    Args:
        input_dimensions: shape of the input tensor
        latent_features: number of bottleneck features
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_features: int = 220,
    ):
        super().__init__()

        blocks: List[nn.Module] = []
        for index, out_channels in enumerate((60, 120, 240, 480, 960)):
            if index == 0:
                block = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3),
                    _channel_norm_2d(out_channels, affine=True),
                    nn.ReLU(),
                )
            else:
                block = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    _channel_norm_2d(out_channels, affine=True),
                    nn.ReLU(),
                )

            in_channels = out_channels
            blocks += [block]

        blocks += [nn.Conv2d(out_channels, latent_features, kernel_size=3, padding=1)]

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class _ResidualBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.sequence = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride, padding=padding),
            _channel_norm_2d(channels, affine=True),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size, stride, padding=padding),
            _channel_norm_2d(channels, affine=True),
        )

    def forward(self, x):
        return x + self.sequence(x)


class HiFiCGenerator(torch.nn.Module):
    """
    High-Fidelity Generative Image Compression (HiFiC) generator.

    Args:
        input_dimensions: shape of the input tensor
        batch_size: number of images per batch
        latent_features: number of bottleneck features
        n_residual_blocks: number of residual blocks
    """

    def __init__(
        self,
        image_channels: int = 3,
        latent_features: int = 220,
        n_residual_blocks: int = 9,
    ):
        super(HiFiCGenerator, self).__init__()

        self.n_residual_blocks = n_residual_blocks

        filters = [960, 480, 240, 120, 60]

        self.block_0 = nn.Sequential(
            _channel_norm_2d(latent_features, affine=True),
            nn.Conv2d(latent_features, filters[0], kernel_size=3, padding=1),
            _channel_norm_2d(filters[0], affine=True),
        )

        resid_blocks = []
        for _ in range(self.n_residual_blocks):
            resid_blocks.append(_ResidualBlock((filters[0])))

        self.resid_blocks = nn.Sequential(*resid_blocks)

        blocks: List[nn.Module] = []
        in_channels = filters[0]
        for out_channels in filters[1:]:
            blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        output_padding=1,
                        stride=2,
                        padding=1,
                    ),
                    _channel_norm_2d(out_channels, affine=True),
                    nn.ReLU(),
                )
            )

            in_channels = out_channels

        blocks.append(
            nn.Conv2d(
                filters[-1], out_channels=image_channels, kernel_size=7, padding=3
            )
        )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        x = self.block_0(x)
        x = x + self.resid_blocks(x)

        return self.blocks(x)
