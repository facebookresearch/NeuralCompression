# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing

import torch


def estimate_tails(
    func: typing.Callable[[torch.Tensor], torch.Tensor],
    target: float,
    shape: int,
    device: typing.Union[torch.device, str, None] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Estimates approximate tail quantiles.

    A simple Adam iteration is ran to determine tail quantiles. The objective
    is to find an :math:`x` such that :math:`func(x) = target`. For instance,
    if :math:`func` is a CDF and :math:`target` is a quantile value, this would
    find the approximate location of that quantile. Note that :math:`func` is
    assumed to be monotonic.

    When each tail estimate has passed the optimal value of :math:`x`, the
    algorithm does 100 additional iterations and then stops. This operation is
    vectorized. The tensor shape of :math:`x` is given by :math:`shape`, and
    :math:`target` must have a shape that is broadcastable to the output of
    :math:`func(x)`.

    Args:
        func: a function that computes the cumulative distribution function,
            survival function, or similar.
        target: desired target value.
        shape: shape representing :math:`x`.
        device: PyTorch device.
        dtype: PyTorch dtype of the computation and the return value.

    Returns:
        the solution, :math:`x`.
    """
    if not device:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    eps = torch.finfo(torch.float32).eps

    counts = torch.zeros(shape, dtype=torch.int32)

    tails = torch.zeros(shape, device=device, dtype=dtype, requires_grad=True)

    mean = torch.zeros(shape, dtype=dtype)

    variance = torch.ones(shape, dtype=dtype)

    while torch.min(counts) < 100:
        abs(func(tails) - target).backward(torch.ones_like(tails))

        gradient = tails.grad.cpu()

        with torch.no_grad():
            mean = 0.9 * mean + (1.0 - 0.9) * gradient

            variance = 0.99 * variance + (1.0 - 0.99) * torch.square(gradient)

            tails -= (1e-2 * mean / (torch.sqrt(variance) + eps)).to(device)

        condition = torch.logical_or(counts > 0, gradient.cpu() * tails.cpu() > 0)

        counts = torch.where(condition, counts + 1, counts)

        tails.grad.zero_()

    return tails
