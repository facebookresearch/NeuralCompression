"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import typing

import torch.nn.functional


def _non_saturating_adversarial_loss(
    authentic_image: typing.Optional[torch.Tensor] = None,
    synthetic_image: typing.Optional[torch.Tensor] = None,
    authentic_predictions: typing.Optional[torch.Tensor] = None,
    synthetic_predictions: typing.Optional[torch.Tensor] = None,
) -> typing.Optional[typing.Tuple[torch.Tensor, torch.Tensor]]:
    if not authentic_predictions or not synthetic_predictions:
        raise ValueError

    authentic_discriminator_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        authentic_predictions, torch.ones_like(authentic_predictions)
    )

    synthetic_discriminator_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        synthetic_predictions, torch.zeros_like(synthetic_predictions)
    )

    discriminator_loss = authentic_discriminator_loss + synthetic_discriminator_loss

    generator_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        synthetic_predictions, torch.ones_like(synthetic_predictions)
    )

    return discriminator_loss, generator_loss
