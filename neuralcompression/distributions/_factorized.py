"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Any, Dict, Optional, Tuple

import torch
from torch import Size, Tensor
from torch.distributions import Distribution
from torch.distributions.constraints import Constraint


class Factorized(Distribution):
    def __init__(
        self,
        num_filters: Tuple[int, ...] = (3, 3),
        init_scale: float = 10.0,
        dtype: torch.dtype = torch.float32,
    ):
        super(Factorized, self).__init__(
            validate_args=False,
        )

        self._num_filters = num_filters

        self._init_scale = init_scale

        self._dtype = dtype

    @property
    def arg_constraints(self) -> Dict[str, Constraint]:
        return {}

    @property
    def mean(self) -> Tensor:
        pass

    @property
    def quantization_offset(self) -> Tensor:
        return torch.tensor(0)

    @property
    def support(self) -> Optional[Any]:
        pass

    @property
    def variance(self) -> Tensor:
        pass

    def cdf(self, value: Tensor) -> Tensor:
        pass

    def entropy(self) -> Tensor:
        pass

    def enumerate_support(self, expand: bool = True) -> Tensor:
        pass

    def expand(self, batch_shape, _instance=None) -> Distribution:
        pass

    def icdf(self, value: Tensor) -> Tensor:
        pass

    def log_cdf(self, value: Tensor) -> Tensor:
        pass

    def log_prob(self, value: Tensor) -> Tensor:
        pass

    def log_survival_function(self, value: Tensor) -> Tensor:
        pass

    def lower_tail(self, tail_mass: float) -> Tensor:
        pass

    def prob(self, value: Tensor) -> Tensor:
        pass

    def rsample(self, sample_shape: Size = Size()):
        pass

    def survival_function(self, value: Tensor) -> Tensor:
        pass

    def upper_tail(self, tail_mass: float) -> Tensor:
        pass
