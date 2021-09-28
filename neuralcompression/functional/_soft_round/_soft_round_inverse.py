import torch


def soft_round_inverse(y: torch.Tensor, alpha: float, eps: float = 1e-3) -> torch.Tensor:
    alpha = torch.tensor(alpha)

    eps = torch.tensor(eps)

    maximum = torch.maximum(alpha, eps)

    m = torch.floor(y) + .5

    x = m + torch.clamp(torch.atanh((y - m) * (torch.tanh(maximum / 2.0) * 2.0)) / maximum, -0.5, 0.5)

    return torch.where(alpha < eps, y, x)
