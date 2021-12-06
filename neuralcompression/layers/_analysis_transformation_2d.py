# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch import Tensor
from torch.nn import Conv2d, Module, Sequential

from ._generalized_divisive_normalization import GeneralizedDivisiveNormalization


class AnalysisTransformation2D(Module):
    def __init__(
        self,
        network_channels: int,
        compression_channels: int,
        in_channels: int = 3,
    ):
        super(AnalysisTransformation2D, self).__init__()

        self.encode = Sequential(
            Conv2d(
                in_channels,
                network_channels,
                (5, 5),
                (2, 2),
                2,
            ),
            GeneralizedDivisiveNormalization(network_channels),
            Conv2d(
                network_channels,
                network_channels,
                (5, 5),
                (2, 2),
                2,
            ),
            GeneralizedDivisiveNormalization(network_channels),
            Conv2d(
                network_channels,
                network_channels,
                (5, 5),
                (2, 2),
                2,
            ),
            GeneralizedDivisiveNormalization(network_channels),
            Conv2d(
                network_channels,
                compression_channels,
                (5, 5),
                (2, 2),
                2,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x:

        Returns:
        """
        return self.encode(x)
