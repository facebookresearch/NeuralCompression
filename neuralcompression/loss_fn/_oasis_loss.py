# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import Tensor

from ._gan_losses import DiscriminatorLoss, GeneratorLoss


def _verify_logit_target_shape(logits: Tensor, target: Tensor):
    if logits.ndim != 4:
        raise ValueError("Only expect 4-dimensional logits.")

    expected_target_numel = logits.shape[0] * logits.shape[-2] * logits.shape[-1]
    actual_target_shape = target.numel()
    logits_shape = logits.shape
    if expected_target_numel != actual_target_shape:
        raise ValueError(
            f"Based on logits size {logits_shape}, expected target numel to be "
            f"{expected_target_numel}, but found {actual_target_shape}"
        )


class OASISDiscriminatorLoss(DiscriminatorLoss):
    """
    Discriminator loss function from OASIS paper.

    This applies a spatially-distributed non-saturating GAN discriminator loss.
    Effectively, it takes a spatially-varying target map and calculates the
    crossentropy over the targets.

    This loss was introduced in the following paper:

    You Only Need Adversarial Supervision for Semantic Image Synthesis
    Vadim Sushko, Edgar Schönfeld, Dan Zhang, Juergen Gall, Bernt Schiele,
    Anna Khoreva
    """

    def __call__(
        self, real_logits: Tensor, fake_logits: Tensor, target: Tensor
    ) -> Tensor:
        """
        Calculate OASIS discriminator loss.

        Args:
            real_logits: The 4-D map of logits for the real variables. The
                dimensions are expected to be ``(batch, num_classes+1, ny, nx).
            fake_logits: the 4-D map of logits for the fake variables, with the
                same dimensions as the ``real_logits`.
            target: A 3-D map of spatially-varying ground truth labels.

        Returns:
            The OASIS discriminator loss value.
        """
        _verify_logit_target_shape(real_logits, target)
        _verify_logit_target_shape(fake_logits, target)
        if target.dtype != torch.long:
            raise ValueError("Expected target to have dtype torch.long.")

        batch_size, num_classes, _, _ = real_logits.shape

        target = target.view(batch_size, -1)
        return 0.5 * (
            F.cross_entropy(real_logits.view(batch_size, num_classes, -1), target + 1)
            + F.cross_entropy(
                fake_logits.view(batch_size, num_classes, -1),
                torch.zeros_like(target),
            )
        )


class OASISGeneratorLoss(GeneratorLoss):
    """
    Generator loss function from OASIS paper.

    This applies a spatially-distributed non-saturating GAN generator loss.
    Effectively, it takes a spatially-varying target map and calculates the
    crossentropy over the targets.

    This loss was introduced in the following paper:

    You Only Need Adversarial Supervision for Semantic Image Synthesis
    Vadim Sushko, Edgar Schönfeld, Dan Zhang, Juergen Gall, Bernt Schiele,
    Anna Khoreva
    """

    def __call__(self, logits: Tensor, target: Tensor) -> Tensor:
        """
        Calculate OASIS discriminator loss.

        Args:
            logits: The 4-D map of logits for the variables. The dimensions are
                expected to be ``(batch, num_classes+1, ny, nx).
            target: A 3-D map of spatially-varying ground truth labels.

        Returns:
            The OASIS generator loss value.
        """
        _verify_logit_target_shape(logits, target)
        if target.dtype != torch.long:
            raise ValueError("Expected target to have dtype torch.long.")

        batch_size, num_classes, _, _ = logits.shape

        target = target.view(batch_size, -1)
        return F.cross_entropy(logits.view(batch_size, num_classes, -1), target + 1)
