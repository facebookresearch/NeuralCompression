# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing

import torch


def _least_squares_adversarial_loss(
    authentic: torch.Tensor,
    synthetic: torch.Tensor,
) -> typing.Optional[typing.Tuple[torch.Tensor, torch.Tensor]]:
    authentic_discriminator_loss = torch.mean(torch.square(authentic - 1.0))
    synthetic_discriminator_loss = torch.mean(torch.square(synthetic))

    discriminator_loss = 0.5 * (
        authentic_discriminator_loss + synthetic_discriminator_loss
    )

    generator_loss = 0.5 * torch.mean(torch.square(synthetic - 1.0))

    return discriminator_loss, generator_loss
