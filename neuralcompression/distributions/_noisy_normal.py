"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from torch import Tensor
from torch.distributions import Normal

from ._uniform_noise import UniformNoise


class NoisyNormal(UniformNoise):
    def __init__(self, **kwargs):
        distribution = Normal(**kwargs)

        super(NoisyNormal, self).__init__(distribution)

    def log_prob(self, value: Tensor) -> Tensor:
        raise NotImplementedError
