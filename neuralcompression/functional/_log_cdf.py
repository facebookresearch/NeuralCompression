"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import Tensor
from torch.distributions import Distribution, Normal

from ._log_ndtr import log_ndtr


def log_cdf(
    x: Tensor,
    distribution: Distribution,
):
    """Logarithm of the distribution’s cumulative distribution function (CDF).

    Args:
        x:
        distribution:

    Returns:
        the log of the area under the distribution’s probability density
            function (PDF), integrated from minus infinity to ``x``.
    """
    if isinstance(distribution, Normal):
        standardized = (x - distribution.loc) / distribution.scale

        return log_ndtr(standardized)

    return torch.log(distribution.cdf(x))
