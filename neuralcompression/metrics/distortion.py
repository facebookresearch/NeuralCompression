# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Optional, Sequence

import torch
from torchmetrics import Metric

from neuralcompression.functional import multiscale_structural_similarity
from neuralcompression.functional._lpips import _load_lpips_model
from neuralcompression.functional._multiscale_structural_similarity import (
    MS_SSIM_FACTORS,
)


class MultiscaleStructuralSimilarity(Metric):
    def __init__(
        self,
        data_range: float = 1.0,
        window_size: int = 11,
        k1: float = 0.01,
        k2: float = 0.03,
        gaussian_std: float = 1.5,
        power_factors: Sequence[float] = MS_SSIM_FACTORS,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        """
        Computes the multi-scale structural similarity index measure.

        Follows the algorithm in the paper: Wang, Zhou, Eero P. Simoncelli,
        and Alan C. Bovik. "Multiscale structural similarity for image
        quality assessment." Signals, Systems and Computers, 2004.
        https://www.cns.nyu.edu/pub/eero/wang03b.pdf

        Args:
            data_range: dynamic range of the input tensors.
            window_size: window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
            gaussian_std: standard deviation of Gaussian filter to use in SSIM
                calculations.
            power_factors: relative importance of each scale; defaults to
                the values proposed in the paper; the length of
                power_factors determines how many scales to consider.
            compute_on_step: see ``torchmetrics.Metric`` documentation.
            dist_sync_on_step: see ``torchmetrics.Metric`` documentation.
            process_group: see ``torchmetrics.Metric`` documentation.
            dist_sync_fn: see ``torchmetrics.Metric`` documentation.
        """

        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.data_range = data_range
        self.window_size = window_size
        self.k1 = k1
        self.k2 = k2
        self.gaussian_std = gaussian_std
        self.power_factors = power_factors

        self.add_state("score_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):

        self.score_sum += multiscale_structural_similarity(
            preds,
            target,
            data_range=self.data_range,
            window_size=self.window_size,
            k1=self.k1,
            k2=self.k2,
            gaussian_std=self.gaussian_std,
            power_factors=self.power_factors,
            reduction="sum",
        )

        self.total += preds.shape[0]

    def compute(self):
        return self.score_sum / self.total


class LearnedPerceptualImagePatchSimilarity(Metric):
    """
    Learned perceptual image patch similarity.

    Implementation of the LPIPS metric introduced in "The Unreasonable
    Effectiveness of Deep Features as a Perceptual Metric" by Richard Zhang,
    Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang.

    NOTE: the input images to this function as assumed to have values
    in the range [-1,1], NOT in the range [0,1]. If image values are in
    the range [0,1], pass ``normalize=True``.

    Args:
        base_network: the pretrained architecture to extract features from.
            Must be one of ``'alex'``, ``'vgg'``, ``'squeeze'``,
            corresponding to AlexNet, VGG, and SqueezeNet, respectively.
        use_linear_calibration: whether to use pretrained weights to
            compute a weighed average across model layers. If ``False``,
            different layers are averaged.
        linear_weights_version: which pretrained linear weights version
            to use. Must be one of ``'0.0'``, ``'0.1'``. This is ignored if
            ``use_linear_calibration==False``.
        normalize: whether to rescale the input images from the
            range [0,1] to the range [-1,1] before computing LPIPS.
        compute_on_step: see ``torchmetrics.Metric`` documentation.
        dist_sync_on_step: see ``torchmetrics.Metric`` documentation.
        process_group: see ``torchmetrics.Metric`` documentation.
        dist_sync_fn: see ``torchmetrics.Metric`` documentation.
    """

    def __init__(
        self,
        base_network: str = "alex",
        use_linear_calibration: bool = True,
        linear_weights_version: str = "0.1",
        normalize: bool = False,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):

        if base_network not in ("alex", "vgg", "squeeze"):
            raise ValueError(
                f"Unknown base network {base_network}"
                " - please pass one of 'alex', 'vgg', or 'squeeze'."
            )

        if linear_weights_version not in ("0.0", "0.1"):
            raise ValueError(
                f"Unknown linear weights version {linear_weights_version}"
                " - please pass '0.0' or '0.1'."
            )

        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.normalize = normalize
        self.model = _load_lpips_model(
            base_network=base_network,
            linear_weights_version=linear_weights_version,
            use_linear_calibration=use_linear_calibration,
        )

        self.add_state("score_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def train(self, _):
        """
        Ensuring that LPIPS model is always in eval mode.
        """
        return super().train(False)

    def update(self, preds, target):
        self.score_sum += self.model(preds, target, normalize=self.normalize).sum()
        self.total += preds.shape[0]

    def compute(self):
        return self.score_sum / self.total
