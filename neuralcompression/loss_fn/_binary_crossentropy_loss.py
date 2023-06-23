# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import Tensor

from ._gan_losses import DiscriminatorLoss, GeneratorLoss


class BinaryCrossentropyDiscriminatorLoss(DiscriminatorLoss):
    """
    Binary cross-entropy discriminator loss function.

    This callable applies the standard non-saturating binary cross-entropy
    loss for training GANs.

    Args:
        real_logits: Logits of the real input.
        fake_logits: Logits of the fake input.
        target: The target for crossentropy. Typically, this will be a ones
            tensor of the same size as ``real_logits``.

    Returns:
        The binary cross-entropy discriminator loss.
    """

    def __call__(
        self, real_logits: Tensor, fake_logits: Tensor, target: Tensor
    ) -> Tensor:
        if target.ndim == 0:
            target = target * torch.ones_like(real_logits)
        elif target.ndim != real_logits.ndim:
            raise ValueError("Misconfigured GAN loss starget dimension.")

        return 0.5 * (
            F.binary_cross_entropy_with_logits(real_logits, target)
            + F.binary_cross_entropy_with_logits(
                fake_logits,
                torch.zeros_like(target),
            )
        )


class BinaryCrossentropyGeneratorLoss(GeneratorLoss):
    """
    Binary cross-entropy generator loss function.

    This callable applies the standard non-saturating binary cross-entropy
    loss for training GANs. It is a sign flip of the discriminator loss term
    and includes no 'fake' logit calculation.

    Args:
        logits: Logits of the input.
        target: The target for crossentropy. Typically, this will be a ones
            tensor of the same size as ``logits``.

    Returns:
        The binary cross-entropy generator loss.
    """

    def __call__(self, logits: Tensor, target: Tensor) -> Tensor:
        if target.ndim == 0:
            target = target * torch.ones_like(logits)
        elif target.ndim != logits.ndim:
            raise ValueError("Misconfigured GAN loss starget dimension.")

        return F.binary_cross_entropy_with_logits(logits, target)
