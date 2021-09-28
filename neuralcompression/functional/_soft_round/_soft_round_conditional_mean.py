import torch

from ._soft_round_inverse import soft_round_inverse


def soft_round_conditional_mean(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return soft_round_inverse(x - 0.5, alpha) + 0.5
