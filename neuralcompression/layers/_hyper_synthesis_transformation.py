"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from torch import Tensor
from torch.nn import Conv2d, ConvTranspose2d, Module, ReLU, Sequential


class HyperSynthesisTransformation(Module):
    def __init__(self, m: int, n: int):
        super(HyperSynthesisTransformation, self).__init__()

        self._modules = Sequential(
            ConvTranspose2d(n, n, (5, 5), (2, 2), (2, 2), (1, 1)),
            ReLU(inplace=True),
            ConvTranspose2d(n, n, (5, 5), (2, 2), (2, 2), (1, 1)),
            ReLU(inplace=True),
            Conv2d(n, m, (3, 3), (1, 1), 1),
            ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._modules(x)
