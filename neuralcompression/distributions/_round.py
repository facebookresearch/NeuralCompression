"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import Tensor
from torch.distributions import Distribution

from ._monotonic import Monotonic
from ..functional import lower_tail, upper_tail


class Round(Monotonic):
    """Adapts a continuous distribution via an ascending monotonic function and
        rounding.

    Args:
        distribution: A `torch.distributions.Distribution` object representing
            a continuous-valued random variable.
    """
    _invertible = False

    def __init__(self, distribution: Distribution):
        super(Round, self).__init__(distribution)

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
