"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from torch import Tensor
from torch.distributions import (
    AffineTransform,
    SigmoidTransform,
    TransformedDistribution,
    Uniform,
)

from ._uniform_noise import UniformNoise


class NoisyLogistic(UniformNoise):
    """Logistic distribution with additive independent and identically
    distributed (i.i.d.) uniform noise.

    The method is described in Appendix 6.2. of:

        | “Variational Image Compression with a Scale Hyperprior”
        | Johannes Ballé, David Minnen, Saurabh Singh, Sung Jin Hwang, Nick Johnston
        | https://arxiv.org/abs/1802.01436
    """

    def __init__(self, **kwargs):
        distribution = TransformedDistribution(
            Uniform(0, 1),
            [SigmoidTransform().inv, AffineTransform(**kwargs)],
        )

        super(NoisyLogistic, self).__init__(distribution)

    def log_prob(self, value: Tensor) -> Tensor:
        raise NotImplementedError
