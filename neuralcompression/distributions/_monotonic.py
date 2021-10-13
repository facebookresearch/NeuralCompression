"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import abc
from abc import ABCMeta
from typing import Dict, Optional

from torch import Size, Tensor
from torch.distributions import Distribution
from torch.distributions.constraints import Constraint

from ..functional import lower_tail, upper_tail


class Monotonic(Distribution, metaclass=ABCMeta):
    """Adapts a continuous distribution via an ascending monotonic function.

    The method is described in Appendix E. of:

    > “Universally Quantized Neural Compression”
    > Eirikur Agustsson, Lucas Theis
    > https://arxiv.org/abs/2006.09952

    Args:
        distribution: A `torch.distributions.Distribution` object representing
            a continuous-valued random variable.
    """

    _invertible: bool = False

    def __init__(self, distribution: Distribution, **kwargs):
        super(Monotonic, self).__init__(**kwargs)

        self._distribution = distribution

    @property
    def arg_constraints(self) -> Dict[str, Constraint]:
        return self._distribution.arg_constraints

    @property
    def mean(self) -> Tensor:
        return self._distribution.mean

    def quantile(self, value: Tensor) -> Tensor:
        if not self._invertible:
            raise NotImplementedError

        return self.transform(
            self._distribution.icdf(value),
        )

    # @property
    # def quantization_offset(self):
    #     if not self._invertible:
    #         raise NotImplementedError
    #
    #     return self.transform(
    #         quantization_offset(
    #             self._distribution,
    #         )
    #     )

    @property
    def support(self) -> Optional[Constraint]:
        return self._distribution.support

    @property
    def variance(self) -> Tensor:
        return self._distribution.variance

    def cdf(self, value: Tensor) -> Tensor:
        return self._distribution.cdf(self.inverse_transform(value))

    def entropy(self) -> Tensor:
        return self._distribution.entropy()

    def enumerate_support(self, expand: bool = True) -> Tensor:
        return self._distribution.enumerate_support(expand)

    def expand(
        self,
        batch_shape: Size,
        _instance: Optional[Distribution] = None,
    ) -> Distribution:
        return self._distribution.expand(
            batch_shape,
            _instance,
        )

    def icdf(self, value: Tensor) -> Tensor:
        return self._distribution.icdf(value)

    @abc.abstractmethod
    def inverse_transform(self, value: Tensor) -> Tensor:
        raise NotImplementedError

    def log_prob(self, value: Tensor) -> Tensor:
        raise NotImplementedError

    def lower_tail(self, tail_mass: float) -> Tensor:
        if not self._invertible:
            raise NotImplementedError

        return self.transform(
            lower_tail(
                self._distribution,
                tail_mass,
            )
        )

    def rsample(self, sample_shape: Size = Size()) -> Tensor:
        return self._distribution.rsample(sample_shape)

    @abc.abstractmethod
    def transform(self, value: Tensor) -> Tensor:
        raise NotImplementedError

    def upper_tail(self, tail_mass: float) -> Tensor:
        if not self._invertible:
            raise NotImplementedError

        return self.transform(
            upper_tail(
                self._distribution,
                tail_mass,
            )
        )
