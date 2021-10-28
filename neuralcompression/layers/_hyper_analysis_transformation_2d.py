"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

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
        in_channels: number of channels in the input signal.
        out_channels: number of channels produced by the transformation.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(HyperAnalysisTransformation2D, self).__init__()

        self.models = Sequential(
            Conv2d(in_channels, out_channels, (3, 3), (1, 2), 2),
            ReLU(inplace=True),
            Conv2d(out_channels, out_channels, (5, 5), (2, 2), 2),
            ReLU(inplace=True),
            Conv2d(out_channels, out_channels, (5, 5), (2, 2), 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.models(x)
