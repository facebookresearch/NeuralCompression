# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing

import torch.nn.functional


def _non_saturating_adversarial_loss(
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
