# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn.functional as F
from torch import Tensor

from ._distortion_loss import DistortionLoss


class MSELoss(DistortionLoss):
    """
    Mean-squared error loss function.

    This function assumes the input images are in the [0.0, 1.0] range. To
    match existing neural compression implementations, the MSE result is
    multiplied by ``scale_param``, which is 255.0**2 by default.

    Args:
        scale_param: The scaling parameter for the loss term.
    """

    def __init__(self, scale_param: float = 255.0**2):
        super().__init__()
        self.scale_param = scale_param

    def __call__(self, image: Tensor, reference: Tensor) -> Tensor:
        """
        Apply the MSE loss.

        Args:
            image: The image to measure MSE loss from.
            reference: The reference or 'ground truth' image.

        Returns:
            The MSE loss value.
        """
        return self.scale_param * F.mse_loss(image, reference)
