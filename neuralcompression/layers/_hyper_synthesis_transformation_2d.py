"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

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
        n: number of channels in the input signal.
        m: number of inferred latent features.
    """

    def __init__(
        self,
        n: int,
        m: int,
        activation: Module = ReLU(inplace=True),
        convolution: Optional[Module] = None,
    ):
        super(HyperSynthesisTransformation2D, self).__init__()

        if not convolution:
            convolution = Conv2d(n, m, (3, 3), (1, 1), (1, 1))

        self.sequence = Sequential(
            ConvTranspose2d(n, n, (5, 5), (2, 2), (2, 2), (1, 1)),
            activation,
            ConvTranspose2d(n, n, (5, 5), (2, 2), (2, 2), (1, 1)),
            activation,
            convolution,
            activation,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.sequence(x)
