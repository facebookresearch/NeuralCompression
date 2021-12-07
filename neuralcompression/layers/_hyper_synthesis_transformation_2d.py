# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from torch import Tensor
from torch.nn import Conv2d, ConvTranspose2d, Module, ReLU, Sequential


class HyperSynthesisTransformation2D(Module):
    """Applies the 2D hyper synthesis transformation over an input signal.

    The hyper synthesis transformation is used to infer the latent
    representation of an input signal.

    The method is described in:

        | “Variational Image Compression with a Scale Hyperprior”
        | Johannes Ballé, David Minnen, Saurabh Singh, Sung Jin Hwang,
            Nick Johnston
        | https://arxiv.org/abs/1802.01436

    Args:
        network_channels: number of channels in the input signal.
        compression_channels: number of inferred latent features.
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
        super(HyperSynthesisTransformation2D, self).__init__()

        if activation is None:
            activation = ReLU(inplace=True)

        self.decode = Sequential(
            ConvTranspose2d(
                network_channels,
                network_channels,
                (5, 5),
                (2, 2),
                (5 // 2, 5 // 2),
                (1, 1),
            ),
            activation,
            ConvTranspose2d(
                network_channels,
                network_channels,
                (5, 5),
                (2, 2),
                (5 // 2, 5 // 2),
                (1, 1),
            ),
            activation,
            Conv2d(
                network_channels,
                compression_channels,
                (in_channels, in_channels),
                (1, 1),
                (in_channels // 2, in_channels // 2),
            ),
            activation,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(x)
