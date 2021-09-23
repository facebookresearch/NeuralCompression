import typing

import torch
import torch.nn.functional


def _non_saturating_loss(
        target_a: torch.Tensor,
        target_b: torch.Tensor,
        input_a: typing.Optional[torch.Tensor] = None,
        input_b: typing.Optional[torch.Tensor] = None,
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    da = torch.nn.functional.binary_cross_entropy_with_logits(target_a, torch.ones_like(target_a))
    db = torch.nn.functional.binary_cross_entropy_with_logits(target_b, torch.zeros_like(target_b))

    return da + db, (torch.nn.functional.binary_cross_entropy_with_logits(target_b, torch.ones_like(target_b)))
