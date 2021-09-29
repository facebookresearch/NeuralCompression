import typing

import torch


def estimate_tails(
    cdf: typing.Callable[[torch.Tensor], torch.Tensor],
    target: float,
    shape: int,
    device: typing.Optional[typing.Union[int, str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
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
        abs(cdf(tails) - target).backward(torch.ones_like(tails))

        gradient = tails.grad.cpu()

        with torch.no_grad():
            mean = 0.9 * mean + (1.0 - 0.9) * gradient

            variance = 0.99 * variance + (1.0 - 0.99) * torch.square(gradient)

            tails -= (1e-2 * mean / (torch.sqrt(variance) + eps)).to(device)

        condition = torch.logical_or(counts > 0, gradient.cpu() * tails.cpu() > 0)

        counts = torch.where(condition, counts + 1, counts)

        tails.grad.zero_()

    return tails
