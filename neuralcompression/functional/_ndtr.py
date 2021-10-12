import math

import torch
from torch import Tensor


def ndtr(x: Tensor) -> Tensor:
    x *= math.sqrt(0.5)

    y = 0.5 * torch.erfc(abs(x))

    return torch.where(
        abs(x) < math.sqrt(0.5),
        0.5 + 0.5 * torch.erf(x),
        torch.where(
            x > 0,
            1 - y,
            y,
        ),
    )
