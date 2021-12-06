# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor


def log_expm1(
    x: Tensor,
) -> Tensor:
    r"""Logarithm of :math:`e^{x} - 1`.

    If :math:`x` is large, ``torch.exp(x)`` returns ``float("inf")`` whereas:

    .. math::
        \log e^{x} - 1 \approx x.

    Therefore, an approximation for :math:`x > 15` is used, such that the
    output is not infinity for all positive values of :math:`x`.

    Args:
        x: the input tensor.

    Returns:
        the logarithm of :math:`e^{x} - 1`.
    """
    return torch.where(
        x < 15.0,
        torch.log(torch.expm1(torch.minimum(x, torch.tensor(15.0)))),
        x,
    )
