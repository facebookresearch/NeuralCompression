import typing

import torch.nn.functional


def non_saturating_adversarial_loss(
    authentic: torch.Tensor,
    synthetic: torch.Tensor,
) -> typing.Optional[typing.Tuple[torch.Tensor, torch.Tensor]]:
    authentic_discriminator_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        authentic, torch.ones_like(authentic)
    )

    synthetic_discriminator_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        synthetic, torch.zeros_like(synthetic)
    )

    discriminator_loss = authentic_discriminator_loss + synthetic_discriminator_loss

    generator_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        synthetic, torch.ones_like(synthetic)
    )

    return discriminator_loss, generator_loss
