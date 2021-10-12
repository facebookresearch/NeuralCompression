"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from torch import Tensor
from torch.distributions import Distribution

from ._monotonic import Monotonic
from ..functional import soft_round, soft_round_inverse


class SoftRound(Monotonic):
    def __init__(self, distribution: Distribution, alpha: float):
        super(SoftRound, self).__init__(distribution)

        self._alpha = alpha

    def inverse_transform(self, value: Tensor) -> Tensor:
        return soft_round_inverse(value, self._alpha)

    def log_prob(self, value: Tensor) -> Tensor:
        raise NotImplementedError

    def transform(self, value: Tensor) -> Tensor:
        return soft_round(value, self._alpha)
