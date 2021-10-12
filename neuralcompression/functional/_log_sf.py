"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from torch import Tensor, log1p, no_grad
from torch.distributions import Distribution


def logsf(
    value: Tensor,
    distribution: Distribution,
):
    """Log of the survival function of ``x`` for ``distribution``.

    The log of the survival function is defined as :math:`1 - CDF`.

    Args:
        value:
        distribution: a parameterizable probability distribution from
            ``torch.distributions``, e.g. ``torch.distributions.Categorical``
            or ``torch.distributions.Normal``.

    Returns:
        Log of the survival function.
    """
    with no_grad():
        return log1p(distribution.cdf(value))
