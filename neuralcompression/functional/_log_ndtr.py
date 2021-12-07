# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from torch import Tensor

from ._ndtr import ndtr


def _log_ndtr_asymptotic_series(x: Tensor, series_order: int = 3) -> Tensor:
    t1 = -0.5 * (math.log(2 * math.pi) + x ** 2) - torch.log(-x)
    t2 = torch.zeros_like(x)
    value_even_power = (x ** 2).clone()
    double_fac = 1
    multiplier = -1

    for n in range(1, series_order + 1):
        t2.add_(multiplier * double_fac / value_even_power)
        value_even_power.mul_(x ** 2)
        double_fac *= 2 * n - 1
        multiplier *= -1

    return t1 + torch.log1p(t2)


def log_ndtr(x: Tensor) -> Tensor:
    """Logarithm of the normal cumulative distribution function (CDF).

    Args:
        x: the input tensor.

    Returns:
        the log of the area under the standard Normal probability density
        function (PDF), integrated from minus infinity to :math:`x`.
    """
    if x.dtype == torch.float32:
        m, n = -10.0, 5.0
    elif x.dtype == torch.float64:
        m, n = -20.0, 8.0
    else:
        raise TypeError(
            f"`{log_ndtr.__name__}` doesn’t support {x.dtype}",
        )

    return torch.where(
        x > n,
        torch.log1p(-ndtr(-x)),
        torch.where(
            x >= m,
            torch.log(ndtr(x)),
            _log_ndtr_asymptotic_series(x),
        ),
    )
