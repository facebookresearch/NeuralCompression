# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch import Tensor
from torch.nn import Conv2d, Module, Sequential

from ._generalized_divisive_normalization import GeneralizedDivisiveNormalization


class AnalysisTransformation2D(Module):
    """Applies the 2D analysis transformation over an input signal.

    The analysis transformation is used to generate a reconstructed signal from
    a latent representation.

    The method is described in:

        | End-to-end Optimized Image Compression
        | Johannes Ballé, Valero Laparra, Eero P. Simoncelli
        | https://arxiv.org/abs/1611.01704

    Args:
        network_channels: number of channels in the input signal.
        compression_channels: number of inferred latent features.
        in_channels: number of channels in the input image.
    """

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
        return self.encode(x)
