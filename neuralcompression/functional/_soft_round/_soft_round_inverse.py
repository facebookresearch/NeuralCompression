import torch


def soft_round_inverse(
    y: torch.Tensor, alpha: float, eps: float = 1e-3
) -> torch.Tensor:
    """Inverse of ``soft_round``.

    This operation is described in Sec. 4.1. in the paper:

    > "Universally Quantized Neural Compression"<br />
    > Eirikur Agustsson & Lucas Theis<br />
    > https://arxiv.org/abs/2006.09952

    Args:
        y:
        alpha: smoothness of the approximation
        eps: threshold below which ``soft_round`` returns the identity
    """
    maximum = torch.tensor(max(alpha, eps))

    m = torch.floor(y) + 0.5

    s = (y - m) * (torch.tanh(maximum / 2.0) * 2.0)

    r = torch.atanh(s) / maximum

    r = torch.clamp(r, -0.5, 0.5)

    x = m + r

    return torch.where(torch.tensor(alpha < eps), y, x)
