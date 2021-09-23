import typing

import torch

from ._least_squares_loss import _least_squares_loss
from ._non_saturating_loss import _non_saturating_loss


def generative_loss(
        input: typing.Tuple[torch.Tensor, torch.Tensor],
        target: typing.Tuple[torch.Tensor, torch.Tensor],
        loss_function: str = "non_saturating_loss",
        mode: str = "generator"
) -> torch.Tensor:
    if loss_function == "least_squares_loss":
        d, g = _least_squares_loss(*input, *target)
    elif loss_function == "non_saturating_loss":
        d, g = _non_saturating_loss(*target, *input)
    else:
        raise ValueError

    if mode == "discriminator":
        return d
    elif mode == "generator":
        return g
    else:
        raise ValueError
