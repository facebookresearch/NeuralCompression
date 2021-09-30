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
        device: PyTorch device
        dtype: PyTorch ``dtype`` of the computation and the return value

    Returns:
        the solution, ``x``
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
