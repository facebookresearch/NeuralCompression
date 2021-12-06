# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing

import torch.nn

from ._channel_norm_2d import _channel_norm_2d


class HiFiCEncoder(torch.nn.Module):
    """
    High-Fidelity Generative Image Compression (HiFiC) encoder.

    Args:
        input_dimensions: shape of the input tensor
        latent_features: number of bottleneck features
    """

    def __init__(
        self,
        input_dimensions: typing.Tuple[int, int, int] = (3, 256, 256),
        latent_features: int = 220,
    ):
        super(HiFiCEncoder, self).__init__()

        self.blocks = []

        in_channels = input_dimensions[0]

        out_channels = None

        for index, out_channels in enumerate((60, 120, 240, 480, 960)):
            if index == 0:
                block = torch.nn.Sequential(
                    torch.nn.ReflectionPad2d(3),
                    torch.nn.Conv2d(in_channels, out_channels, (7, 7), (1, 1)),
                    _channel_norm_2d(
                        out_channels,
                        affine=True,
                        momentum=0.1,
                    ),
                    torch.nn.ReLU(),
                )
            else:
                block = torch.nn.Sequential(
                    torch.nn.ReflectionPad2d((0, 1, 1, 0)),
                    torch.nn.Conv2d(
                        in_channels,
                        out_channels,
                        (3, 3),
                        (2, 2),
                        padding=0,
                        padding_mode="reflect",
                    ),
                    _channel_norm_2d(out_channels, affine=True, momentum=0.1),
                    torch.nn.ReLU(),
                )

            in_channels = out_channels

            self.blocks += [block]

        self.blocks += [
            torch.nn.Sequential(
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(out_channels, latent_features, (3, 3), (1, 1)),
            )
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)

        return x
