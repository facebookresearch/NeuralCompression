import typing

import torch
import torch.nn.functional


def _non_saturating_loss(a: torch.Tensor, b: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    da = torch.nn.functional.binary_cross_entropy_with_logits(a, torch.ones_like(a))
    db = torch.nn.functional.binary_cross_entropy_with_logits(b, torch.zeros_like(b))

    return da + db, (torch.nn.functional.binary_cross_entropy_with_logits(b, torch.ones_like(b)))
