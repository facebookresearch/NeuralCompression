# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional

import torch
from torch import Size, Tensor
from torch.distributions import Distribution, Uniform
from torch.distributions.constraints import Constraint

import neuralcompression.functional as ncF


class UniformNoise(Distribution):
    r"""Adapts a continuous distribution via additive identically distributed
    (i.i.d.) uniform noise.

    The provided distribution is modeled after the addition of independent
    uniform noise. Effectively, the base density function is convolved with a
    box kernel of width one. The resulting density can be efficiently evaluated
    via the relation:

    .. math::
        p_{\widetilde{y}}(\widetilde{y}) =
            c(\widetilde{y} + \frac{1}{2}) - c(\widetilde{y} - \frac{1}{2})

    where :math:`p` and :math:`\widetilde{y}` are the base density and the
    unit-width uniform density, respectively, and :math:`c` is the cumulative
    distribution function (CDF) of :math:`p`.

    The method is described in Appendix 6.2. of:

        | “Variational Image Compression with a Scale Hyperprior”
        | Johannes Ballé, David Minnen, Saurabh Singh, Sung Jin Hwang,
            Nick Johnston
        | https://arxiv.org/abs/1802.01436

    Args:
        distribution: an object representing a continuous-valued random
            variable.
    """

    def __init__(self, distribution: Distribution):
        super(UniformNoise, self).__init__(
            distribution.event_shape,
            distribution.batch_shape,
        )

        self._distribution = distribution

        self._uniform_distribution = Uniform(-0.5, 0.5)

    @property
    def arg_constraints(self) -> Dict[str, Constraint]:
        return {}

    @property
    def mean(self) -> Tensor:
        return self._distribution.mean

    @property
    def quantization_offset(self) -> Tensor:
        return ncF.quantization_offset(self._distribution)

    @property
    def support(self) -> Optional[Any]:
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

    def expand(self, batch_shape, _instance=None) -> Distribution:
        return self._distribution.expand(batch_shape)

    def icdf(self, value: Tensor) -> Tensor:
        return self._distribution.icdf(value)

    def log_cdf(self, value: Tensor) -> Tensor:
        return ncF.log_cdf(value, self._distribution)

    def log_prob(self, value: Tensor) -> Tensor:
        log_survival_function_positive = self.log_survival_function(value + 0.5)
        log_survival_function_negative = self.log_survival_function(value - 0.5)

        log_cdf_positive = self.log_cdf(value + 0.5)
        log_cdf_negative = self.log_cdf(value - 0.5)

        condition = log_survival_function_positive < log_cdf_positive

        a = torch.where(
            condition,
            log_survival_function_negative,
            log_cdf_positive,
        )

        b = torch.where(
            condition,
            log_survival_function_positive,
            log_cdf_negative,
        )

        return torch.log1p(-torch.exp(b - a)) + a

    def log_survival_function(self, value: Tensor) -> Tensor:
        return ncF.log_survival_function(value, self._distribution)

    def lower_tail(self, tail_mass: float) -> Tensor:
        return ncF.lower_tail(self._distribution, tail_mass)

    def prob(self, value: Tensor) -> Tensor:
        survival_function_positive = self.survival_function(value + 0.5)
        survival_function_negative = self.survival_function(value - 0.5)

        cdf_positive = self.cdf(value + 0.5)
        cdf_negative = self.cdf(value - 0.5)

        return torch.where(
            survival_function_positive < cdf_positive,
            survival_function_negative - survival_function_positive,
            cdf_positive - cdf_negative,
        )

    def rsample(self, sample_shape: Size = Size()):
        a = self._distribution.sample(sample_shape)
        b = self._uniform_distribution.sample(sample_shape)

        return a + b

    def survival_function(self, value: Tensor) -> Tensor:
        return ncF.survival_function(value, self._distribution)

    def upper_tail(self, tail_mass: float) -> Tensor:
        return ncF.upper_tail(self._distribution, tail_mass)
