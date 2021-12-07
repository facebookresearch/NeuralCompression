# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

from torch import Tensor
from torch.distributions import Normal

from ._uniform_noise import UniformNoise


class NoisyNormal(UniformNoise):
    r"""Normal distribution with additive identically distributed (i.i.d.)
    uniform noise.

    The method is described in Appendix 6.2. of:

        | “Variational Image Compression with a Scale Hyperprior”
        | Johannes Ballé, David Minnen, Saurabh Singh, Sung Jin Hwang,
            Nick Johnston
        | https://arxiv.org/abs/1802.01436

    Args:
        loc: mean of the distribution.
        scale: standard deviation of the distribution.
    """

    def __init__(
        self,
        loc: Union[float, Tensor],
        scale: Union[float, Tensor],
    ):
        distribution = Normal(loc, scale)

        super(NoisyNormal, self).__init__(distribution)
