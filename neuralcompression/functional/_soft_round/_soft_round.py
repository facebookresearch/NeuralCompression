import torch


def soft_round(x: torch.Tensor, alpha: float, eps: float = 1e-3) -> torch.Tensor:
    maximum = torch.tensor(max(alpha, eps))

    m = torch.floor(x) + 0.5

    z = torch.tanh(maximum / 2.0) * 2.0

    y = m + torch.tanh(maximum * (x - m)) / z

    return torch.where(torch.tensor(alpha < eps), x, y)
