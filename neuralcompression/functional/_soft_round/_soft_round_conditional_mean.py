"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch

from ._soft_round_inverse import soft_round_inverse


def soft_round_conditional_mean(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """Conditional mean of ``x`` given noisy soft rounded values.

    Computes:

            g(z) = E[Y | s(Y) + U = z]

    where :math:`s` is the soft-rounding function, :math:`U` is uniform between
    :math:`-0.5` and :math:`0.5` and :math:`Y` is considered uniform when
    truncated to the interval :math:`[z-0.5, z+0.5]`.

    The method is described in Section 4.1. of:

        | “Universally Quantized Neural Compression”
        | Eirikur Agustsson, Lucas Theis
        | https://arxiv.org/abs/2006.09952

    Args:
        x: The input tensor.
        alpha: smoothness of the ``soft_round`` approximation

    Returns:
        The conditional mean, of same shape as ``x``.
    """
    return soft_round_inverse(x - 0.5, alpha) + 0.5
