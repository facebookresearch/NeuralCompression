# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor
from torch.distributions import Distribution, Normal

from ._log_ndtr import log_ndtr


def log_survival_function(
    x: Tensor,
    distribution: Distribution,
) -> Tensor:
    """Logarithm of :math:`x` for a distribution’s survival function.

    Args:
        x: the input tensor.
        distribution: an object representing a continuous-valued random
            variable.

    Returns:
        the logarithm of :math:`x` for a distribution’s survival function.
    """

    if isinstance(distribution, Normal):
        standardized = torch.div((x - distribution.loc), distribution.scale)

        return log_ndtr(-standardized)

    return torch.log1p(distribution.cdf(-x))
