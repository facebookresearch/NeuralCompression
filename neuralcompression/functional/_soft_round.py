# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


def soft_round(x: torch.Tensor, alpha: float, eps: float = 1e-3) -> torch.Tensor:
    """Differentiable approximation of ``torch.round``.

    A larger ``alpha`` correspond to a closer approximation of ``round``. If
        ``alpha`` is close to zero, this function reduces to the identity.

    This operation is described in Sec. 4.1. in the paper:

    > "Universally Quantized Neural Compression"<br />
    > Eirikur Agustsson & Lucas Theis<br />
    > https://arxiv.org/abs/2006.09952

    Args:
        x: The input tensor.
        alpha: smoothness of the approximation
        eps: threshold below which ``soft_round`` returns the identity

    Returns:
        The differentiable approximation of ``torch.round``.
    """
    maximum = torch.tensor(max(alpha, eps))

    m = torch.floor(x) + 0.5

    z = torch.tanh(maximum / 2.0) * 2.0

    y = m + torch.tanh(maximum * (x - m)) / z

    return torch.where(torch.tensor(alpha < eps), x, y)
