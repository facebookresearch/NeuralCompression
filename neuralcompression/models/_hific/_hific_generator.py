# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing

import torch
import torch.nn
import torch.nn.functional

from ._channel_norm_2d import _channel_norm_2d


class _ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        input_dimensions,
        kernel_size=(3, 3),
        stride=(1, 1),
    ):
        super(_ResidualBlock, self).__init__()

        self.activation = torch.nn.ReLU()

        self.pad = torch.nn.ReflectionPad2d(int((kernel_size[0] - 1) / 2))

        self.conv_a = torch.nn.Conv2d(
            input_dimensions[1], input_dimensions[1], kernel_size, stride
        )
        self.conv_b = torch.nn.Conv2d(
            input_dimensions[1], input_dimensions[1], kernel_size, stride
        )

        self.norm_a = _channel_norm_2d(input_dimensions[1], affine=True, momentum=0.1)

        self.norm_b = _channel_norm_2d(input_dimensions[1], affine=True, momentum=0.1)

    def forward(self, x):
        identity_map = x

        features = self.pad(x)

        features = self.conv_a(features)
        features = self.norm_a(features)

        features = self.activation(features)

        features = self.pad(features)

        features = self.conv_b(features)
        features = self.norm_b(features)

        return torch.add(features, identity_map)


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
        input_dimensions: typing.Tuple[int, int, int] = (3, 256, 256),
        batch_size: int = 8,
        latent_features: int = 220,
        n_residual_blocks: int = 9,
    ):
        super(HiFiCGenerator, self).__init__()

        self.n_residual_blocks = n_residual_blocks

        filters = [960, 480, 240, 120, 60]

        self.block_0 = torch.nn.Sequential(
            _channel_norm_2d(
                latent_features,
                affine=True,
                momentum=0.1,
            ),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(latent_features, filters[0], (3, 3), (1, 1)),
            _channel_norm_2d(
                filters[0],
                affine=True,
                momentum=0.1,
            ),
        )

        for m in range(self.n_residual_blocks):
            residual_block_m = _ResidualBlock(
                (batch_size, filters[0], *input_dimensions[1:])
            )

            self.add_module(f"_ResidualBlock_{str(m)}", residual_block_m)

        self.blocks = []

        in_channels = filters[0]

        for out_channels in filters[1:]:
            block = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    (3, 3),
                    (2, 2),
                    (1, 1),
                    (1, 1),
                ),
                _channel_norm_2d(
                    out_channels,
                    affine=True,
                    momentum=0.1,
                ),
                torch.nn.ReLU(),
            )

            self.blocks += [block]

            in_channels = out_channels

        block = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(filters[-1], 3, (7, 7), (1, 1)),
        )

        self.blocks += [block]

    def forward(self, x):
        block_0 = self.block_0(x)

        for m in range(self.n_residual_blocks):
            residual_block_m = getattr(self, f"_ResidualBlock_{str(m)}")

            if m == 0:
                x = residual_block_m(block_0)
            else:
                x = residual_block_m(x)

        x += block_0

        for block in self.blocks:
            x = block(x)

        return x
