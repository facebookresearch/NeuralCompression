"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
from torch import Tensor
from torch.distributions import Distribution, Normal

from neuralcompression.functional import ndtr


def sf(
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
        standardized = torch.div((x - distribution.loc), distribution.scale)

        return ndtr(-standardized)

    return 1.0 - distribution.cdf(x)

