# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch import Tensor
from torch.nn import ConvTranspose2d, Module, Sequential

from ._generalized_divisive_normalization import GeneralizedDivisiveNormalization


class SynthesisTransformation2D(Module):
    def __init__(self, n: int, m: int, in_channels: int = 3):
        super(SynthesisTransformation2D, self).__init__()

        self.model = Sequential(
            ConvTranspose2d(m, n, (5, 5), (2, 2), (2, 2), (1, 1)),
            GeneralizedDivisiveNormalization(n, inverse=True),
            ConvTranspose2d(n, n, (5, 5), (2, 2), (2, 2), (1, 1)),
            GeneralizedDivisiveNormalization(n, inverse=True),
            ConvTranspose2d(n, n, (5, 5), (2, 2), (2, 2), (1, 1)),
            GeneralizedDivisiveNormalization(n, inverse=True),
            ConvTranspose2d(n, in_channels, (5, 5), (2, 2), (2, 2), (1, 1)),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
