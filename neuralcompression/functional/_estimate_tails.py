"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Callable

from torch import (
    Tensor,
    cuda,
    finfo,
    float32,
    int32,
    logical_or,
    no_grad,
    ones,
    ones_like,
    sqrt,
    square,
    where,
    zeros,
)


def estimate_tails(
    func: Callable[[Tensor], Tensor],
    target: float,
    shape: int,
    **kwargs,
) -> Tensor:
    """Estimates approximate tail quantiles.

    This runs a simple Adam iteration to determine tail quantiles. The
        objective is to find an ``x`` such that:

    ```
    func(x) == target
    ```

    For instance, if ``func`` is a CDF and the target is a quantile value, this
        would find the approximate location of that quantile. Note that
        ``func`` is assumed to be monotonic. When each tail estimate has passed
        the optimal value of ``x``, the algorithm does 100 additional
        iterations and then stops. This operation is vectorized. The tensor
        shape of ``x`` is given by `shape`, and `target` must have a shape that
        is broadcastable to the output of ``func(x)``.

    Args:
        func: a function that computes cumulative distribution function,
            survival function, or similar
        target: desired target value
        shape: shape representing ``x``

    Returns:
        the approximate tail quantiles
    """
    if "device" not in kwargs:
        if cuda.is_available():
            kwargs["device"] = "cuda"
        else:
            kwargs["device"] = "cpu"

    if "dtype" not in kwargs:
        kwargs["dtype"] = float32

    eps = finfo(float32).eps

    counts = zeros(shape, dtype=int32)

    tails = zeros(
        shape,
        device=kwargs["device"],
        dtype=kwargs["dtype"],
        requires_grad=True,
    )

    mean = zeros(shape, dtype=kwargs["dtype"])

    variance = ones(shape, dtype=kwargs["dtype"])

    while min(counts) < 100:
        abs(func(tails) - target).backward(ones_like(tails))

        gradient = tails.grad.cpu()

        with no_grad():
            mean = 0.9 * mean + (1.0 - 0.9) * gradient

            variance = 0.99 * variance + (1.0 - 0.99) * square(gradient)

            tails -= (1e-2 * mean / (sqrt(variance) + eps)).to(kwargs["device"])

        condition = logical_or(counts > 0, gradient.cpu() * tails.cpu() > 0)

        counts = where(condition, counts + 1, counts)

        tails.grad.zero_()

    return tails
