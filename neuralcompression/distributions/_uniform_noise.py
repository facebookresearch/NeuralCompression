"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from abc import ABCMeta
from typing import Dict, Optional

from torch import Size, Tensor
from torch.distributions import Distribution
from torch.distributions.constraints import Constraint

from ..functional import lower_tail, upper_tail


class UniformNoise(Distribution, metaclass=ABCMeta):
    """Adapts a continuous distribution via an additive identically distributed
    (i.i.d.) uniform noise.

    The method is described in Appendix 6.2. of:

        | “Variational Image Compression with a Scale Hyperprior”
        | Johannes Ballé, David Minnen, Saurabh Singh, Sung Jin Hwang,
        Nick Johnston
        | https://arxiv.org/abs/1802.01436

    Args:
        distribution: an object representing a continuous-valued random
            variable.
    """

    def __init__(self, distribution: Distribution, **kwargs):
        super(UniformNoise, self).__init__(**kwargs)

        self._distribution = distribution

    @property
    def arg_constraints(self) -> Dict[str, Constraint]:
        return self._distribution.arg_constraints

    @property
    def mean(self) -> Tensor:
        return self._distribution.mean

    # @property
    # def quantization_offset(self) -> Tensor:
    #     return quantization_offset(self._distribution)

    @property
    def support(self) -> Optional[Constraint]:
        return self._distribution.support

    @property
    def variance(self) -> Tensor:
        return self._distribution.variance

    def cdf(self, value: Tensor) -> Tensor:
        return self._distribution.cdf(value)

    def entropy(self) -> Tensor:
        return self._distribution.entropy()

    def enumerate_support(self, expand: bool = True) -> Tensor:
        return self._distribution.enumerate_support(expand)

    def expand(
        self,
        batch_shape: Size,
        _instance: Optional[Distribution] = None,
    ) -> Distribution:
        return self._distribution.expand(batch_shape, _instance)

    def icdf(self, value: Tensor) -> Tensor:
        return self._distribution.icdf(value)

    def log_prob(self, value: Tensor) -> Tensor:
        raise NotImplementedError

    def lower_tail(self, tail_mass: float) -> Tensor:
        return lower_tail(self._distribution, tail_mass)

    def rsample(self, sample_shape: Size = Size()) -> Tensor:
        return self._distribution.rsample(sample_shape)

    def upper_tail(self, tail_mass: float) -> Tensor:
        return upper_tail(self._distribution, tail_mass)
