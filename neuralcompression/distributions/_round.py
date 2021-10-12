"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import Tensor

from ._monotonic import Monotonic
from ..functional import lower_tail, upper_tail


class Round(Monotonic):
    _invertible = False

    def inverse_transform(self, value: Tensor) -> Tensor:
        return torch.ceil(value) - 0.5

    def log_prob(self, value: Tensor) -> Tensor:
        raise NotImplementedError

    def lower_tail(self, tail_mass: float) -> Tensor:
        return torch.floor(lower_tail(self._distribution, tail_mass))

    def transform(self, value: Tensor) -> Tensor:
        return torch.round(value)

    def upper_tail(self, tail_mass: float) -> Tensor:
        return torch.ceil(upper_tail(self._distribution, tail_mass))
