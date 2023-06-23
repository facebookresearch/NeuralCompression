# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from torch import Tensor


class DiscriminatorLoss(nn.Module):
    """
    Abstract base class for discriminator loss functions.
    """

    def __call__(
        self, real_logits: Tensor, fake_logits: Tensor, target: Tensor
    ) -> Tensor:
        raise NotImplementedError


class GeneratorLoss(nn.Module):
    """
    Abstract base class for generator loss functions.
    """

    def __call__(self, logits: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError
