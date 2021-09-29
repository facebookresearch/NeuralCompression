import torch


def soft_round_inverse(
    y: torch.Tensor, alpha: float, eps: float = 1e-3
) -> torch.Tensor:
    alpha, eps = torch.tensor(alpha), torch.tensor(eps)

    maximum = torch.maximum(alpha, eps)

    m = torch.floor(y) + 0.5

    s = (y - m) * (torch.tanh(maximum / 2.0) * 2.0)

    r = torch.atanh(s) / maximum

    r = torch.clamp(r, -0.5, 0.5)

    x = m + r

    return torch.where(alpha < eps, y, x)
