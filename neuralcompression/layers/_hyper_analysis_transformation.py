"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from torch import Tensor
from torch.nn import Conv2d, Module, ReLU, Sequential


class HyperAnalysisTransformation(Module):
    def __init__(self, m: int, n: int):
        super(HyperAnalysisTransformation, self).__init__()

        self._modules = Sequential(
            Conv2d(m, n, (3,), (1,), 2),
            ReLU(inplace=True),
            Conv2d(n, n, (5,), (2,), 2),
            ReLU(inplace=True),
            Conv2d(n, n, (5,), (2,), 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._modules(x)
