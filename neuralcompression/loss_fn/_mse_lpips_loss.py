# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn.functional as F
from torch import Tensor

from ._distortion_loss import DistortionLoss
from ._normfix_lpips import NormFixLPIPS


class MSELPIPSLoss(DistortionLoss):
    """
    MSE-LPIPS distortion loss.

    This class combines both mean-squared error and LPIPS into a single loss
    function. The final loss is calculated as

    ``mse_param * mse + lpips_param * lpips``

    Args:
        mse_param: A parameter for scaling the MSE term.
        lpips_param: A parameter for scaling the LPIPS term.
        normalize: Whether to normalize inputs to the [-1.0, 1.0] range prior
            to LPIPS calculation.
        backbone: String specifying which classifier model to use as the
            backbone for LPIPS calculation.
    """

    def __init__(
        self,
        mse_param: float = 255.0**2,
        lpips_param: float = 1.0,
        normalize: bool = True,
        backbone: str = "vgg",
    ):
        super().__init__()
        self.mse_param = mse_param
        self.lpips_param = lpips_param
        self.normalize = normalize
        self.lpips_model = NormFixLPIPS(net=backbone).eval()
        for param in self.lpips_model.parameters():
            param.requires_grad_(False)

    def __call__(self, image: Tensor, reference: Tensor) -> Tensor:
        """
        Calculate the MSE-LPIPS loss.

        Args:
            image: The test image to calculate its distortion.
            reference: The 'ground truth' image.

        Returns:
            The MSE-LPIPS loss value.
        """
        return (
            self.mse_param * F.mse_loss(image, reference)
            + self.lpips_param
            * self.lpips_model(image, reference, normalize=self.normalize).mean()
        )
