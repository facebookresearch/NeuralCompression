# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch import Tensor
from torch.nn import Conv2d, Module, Sequential

from ._generalized_divisive_normalization import GeneralizedDivisiveNormalization


class AnalysisTransformation2D(Module):
    def __init__(self, n: int, m: int, in_channels: int = 3):
        super(AnalysisTransformation2D, self).__init__()

        self.sequence = Sequential(
            Conv2d(in_channels, n, (5, 5), (2, 2), 2),
            GeneralizedDivisiveNormalization(n),
            Conv2d(n, n, (5, 5), (2, 2), 2),
            GeneralizedDivisiveNormalization(n),
            Conv2d(n, n, (5, 5), (2, 2), 2),
            GeneralizedDivisiveNormalization(n),
            Conv2d(n, m, (5, 5), (2, 2), 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.sequence(x)
