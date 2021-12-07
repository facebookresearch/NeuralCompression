# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch import Tensor
from torch.nn import ConvTranspose2d, Module, Sequential

from ._generalized_divisive_normalization import GeneralizedDivisiveNormalization


class SynthesisTransformation2D(Module):
    def __init__(
        self,
        network_channels: int,
        compression_channels: int,
        in_channels: int = 3,
    ):
        super(SynthesisTransformation2D, self).__init__()

        self.decode = Sequential(
            ConvTranspose2d(
                compression_channels,
                network_channels,
                (5, 5),
                (2, 2),
                (2, 2),
                (1, 1),
            ),
            GeneralizedDivisiveNormalization(network_channels, inverse=True),
            ConvTranspose2d(
                network_channels,
                network_channels,
                (5, 5),
                (2, 2),
                (2, 2),
                (1, 1),
            ),
            GeneralizedDivisiveNormalization(network_channels, inverse=True),
            ConvTranspose2d(
                network_channels,
                network_channels,
                (5, 5),
                (2, 2),
                (2, 2),
                (1, 1),
            ),
            GeneralizedDivisiveNormalization(network_channels, inverse=True),
            ConvTranspose2d(
                network_channels,
                in_channels,
                (5, 5),
                (2, 2),
                (2, 2),
                (1, 1),
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(x)
