import typing

import torch


def _least_squares_loss(a: torch.Tensor, b: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    dy = 0.5 * torch.mean(torch.square(a - 1.0)) + torch.mean(torch.square(b))
    gy = 0.5 * torch.mean(torch.square(b - 1.0))

    return dy, gy
