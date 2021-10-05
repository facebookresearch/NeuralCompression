"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch

from ._non_saturating_adversarial_loss import _non_saturating_adversarial_loss
from ._least_squares_adversarial_loss import _least_squares_adversarial_loss
from ._discriminator_data import _DiscriminatorData


def _generative_adversarial_loss(
    data: _DiscriminatorData, kind: str = "non_saturating", mode: str = "generator"
) -> torch.Tensor:
    if kind == "least_squares":
        f = _least_squares_adversarial_loss
    elif kind == "non_saturating":
        f = _non_saturating_adversarial_loss
    else:
        raise ValueError

    discriminator_loss, generator_loss = f(
        data.authentic_image,
        data.synthetic_image,
        data.authentic_predictions,
        data.synthetic_predictions,
    )

    if mode == "generator":
        return generator_loss
    else:
        return discriminator_loss
