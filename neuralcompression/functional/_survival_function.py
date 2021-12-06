# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor
from torch.distributions import Distribution, Normal

from ._ndtr import ndtr


def survival_function(
    x: Tensor,
    distribution: Distribution,
) -> Tensor:
    """Survival function of :math:`x`. Generally defined as
    ``1 - distribution.cdf(x)``.

    Args:
        x: the input tensor.
        distribution: an object representing a continuous-valued random
            variable.

    Returns:
        the survival function of :math:`x` for the continuous-valued random
        variable distribution.
    """
    if isinstance(distribution, Normal):
        standardized = torch.div((x - distribution.loc), distribution.scale)

        return ndtr(-standardized)

    return 1.0 - distribution.cdf(x)
