# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch import Tensor
from torch.nn import Conv2d, Module, Sequential

from ._generalized_divisive_normalization import GeneralizedDivisiveNormalization


class AnalysisTransformation2D(Module):
    def __init__(self, m: int, n: int):
        super(AnalysisTransformation2D, self).__init__()

        self.model = Sequential(
            Conv2d(3, m, (5, 5), (2, 2), 2),
            GeneralizedDivisiveNormalization(m),
            Conv2d(m, m, (5, 5), (2, 2), 2),
            GeneralizedDivisiveNormalization(m),
            Conv2d(m, m, (5, 5), (2, 2), 2),
            GeneralizedDivisiveNormalization(m),
            Conv2d(m, n, (5, 5), (2, 2), 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
