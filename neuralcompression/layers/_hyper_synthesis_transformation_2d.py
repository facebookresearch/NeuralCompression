"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

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
        in_channels: number of channels in the input signal.
        features: number of inferred latent features.
    """

    def __init__(self, in_channels: int, features: int):
        super(HyperSynthesisTransformation2D, self).__init__()

        self.model = Sequential(
            ConvTranspose2d(features, features, (5, 5), (2, 2), (2, 2), (1, 1)),
            ReLU(inplace=True),
            ConvTranspose2d(features, features, (5, 5), (2, 2), (2, 2), (1, 1)),
            ReLU(inplace=True),
            Conv2d(features, in_channels, (3, 3), (1, 1), 1),
            ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
