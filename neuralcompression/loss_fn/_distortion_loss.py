# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from torch import Tensor


class DistortionLoss(nn.Module):
    """
    Abstract base class for distortion loss functions.
    """

    def __call__(self, image: Tensor, reference: Tensor) -> Tensor:
        raise NotImplementedError
