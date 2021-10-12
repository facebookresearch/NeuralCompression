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
    """
    Args:
        x:
        distribution:

    Returns:
    """
    if isinstance(distribution, Normal):
        return log_ndtr(-x)

    return torch.log1p(distribution.cdf(-x))
