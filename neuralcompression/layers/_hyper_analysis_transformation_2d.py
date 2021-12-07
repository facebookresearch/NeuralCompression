# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch import Tensor
from torch.nn import Conv2d, Module, ReLU, Sequential


class HyperAnalysisTransformation2D(Module):
    """Applies the 2D hyper analysis transformation over an input signal.

    The hyper analysis transformation is used to generate a reconstructed
    signal from a latent representation.

    The method is described in:

        | “Variational Image Compression with a Scale Hyperprior”
        | Johannes Ballé, David Minnen, Saurabh Singh, Sung Jin Hwang,
            Nick Johnston
        | https://arxiv.org/abs/1802.01436

    Args:
        network_channels: number of channels in the input signal.
        compression_channels: number of channels produced by the transformation.
        in_channels:
        activation:
    """

    def __init__(
        self,
        network_channels: int,
        compression_channels: int,
        in_channels: int = 3,
        activation: Optional[Module] = None,
    ):
        super(HyperAnalysisTransformation2D, self).__init__()

        if activation is None:
            activation = ReLU(inplace=True)

        self.encode = Sequential(
            Conv2d(
                compression_channels,
                network_channels,
                (in_channels, in_channels),
                (1, 1),
                (in_channels // 2, in_channels // 2),
            ),
            activation,
            Conv2d(
                network_channels,
                network_channels,
                (5, 5),
                (2, 2),
                (5 // 2, 5 // 2),
            ),
            activation,
            Conv2d(
                network_channels,
                network_channels,
                (5, 5),
                (2, 2),
                (5 // 2, 5 // 2),
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = torch.abs(x)

        return self.encode(x)
