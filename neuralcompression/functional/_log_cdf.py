"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import Tensor
from torch.distributions import Distribution


def logcdf(
    x: Tensor,
    distribution: Distribution,
):
    """
    Args:
        x:
        distribution:

    Returns:
    """
    with torch.no_grad():
        return torch.log(distribution.cdf(x))
