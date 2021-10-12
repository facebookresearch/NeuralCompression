"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math

import torch
from torch import Tensor


def ndtr(x: Tensor) -> Tensor:
    """Normal cumulative distribution function (CDF).

    Args:
        x:

    Returns:
        the area under the standard Normal probability density function (PDF),
            integrated from minus infinity to ``x``.
    """
    x *= math.sqrt(0.5)

    y = 0.5 * torch.erfc(abs(x))

    return torch.where(
        abs(x) < math.sqrt(0.5),
        0.5 + 0.5 * torch.erf(x),
        torch.where(
            x > 0,
            1 - y,
            y,
        ),
    )