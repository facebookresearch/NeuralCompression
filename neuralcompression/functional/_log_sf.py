"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import Tensor
from torch.distributions import Distribution, Normal

from neuralcompression.functional import log_ndtr


def log_sf(
    x: Tensor,
    distribution: Distribution,
) -> Tensor:
    """Log of the survival function of ``x`` for ``distribution``.

    The log of the survival function is defined as :math:`1 - CDF`.

    Args:
        x:
        distribution: a parameterizable probability distribution from
            ``torch.distributions``, e.g. ``torch.distributions.Categorical``
            or ``torch.distributions.Normal``.

    Returns:
        Log of the survival function.
    """
    if isinstance(distribution, Normal):
        return log_ndtr(-x)

    return torch.log1p(distribution.cdf(x))
