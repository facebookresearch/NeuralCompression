import torch


def soft_round(x: torch.Tensor, alpha: float, eps: float = 1e-3) -> torch.Tensor:
    alpha = torch.tensor(alpha)

    eps = torch.tensor(eps)

    maximum = torch.maximum(alpha, eps)

    m = torch.floor(x) + 0.5

    y = m + torch.tanh(maximum * (x - m)) / (torch.tanh(maximum / 2.0) * 2.0)

    return torch.where(alpha < eps, x, y)
