import typing

import torch


def _least_squares_loss(
        input_a: torch.Tensor,
        input_b: torch.Tensor,
        target_a: typing.Optional[torch.Tensor] = None,
        target_b: typing.Optional[torch.Tensor] = None,
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    dy = 0.5 * torch.mean(torch.square(input_a - 1.0)) + torch.mean(torch.square(input_b))
    gy = 0.5 * torch.mean(torch.square(input_b - 1.0))

    return dy, gy
