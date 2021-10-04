"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import typing

import torch


def _least_squares_adversarial_loss(
    authentic_image: typing.Optional[torch.Tensor] = None,
    synthetic_image: typing.Optional[torch.Tensor] = None,
    authentic_predictions: typing.Optional[torch.Tensor] = None,
    synthetic_predictions: typing.Optional[torch.Tensor] = None,
) -> typing.Optional[typing.Tuple[torch.Tensor, torch.Tensor]]:
    if not authentic_image or not synthetic_image:
        raise ValueError

    authentic_discriminator_loss = torch.mean(torch.square(authentic_image - 1.0))
    synthetic_discriminator_loss = torch.mean(torch.square(synthetic_image))

    discriminator_loss = 0.5 * (
        authentic_discriminator_loss + synthetic_discriminator_loss
    )

    generator_loss = 0.5 * torch.mean(torch.square(synthetic_image - 1.0))

    return discriminator_loss, generator_loss
