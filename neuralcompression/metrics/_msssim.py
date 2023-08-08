# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Optional, Sequence

import torch
from torchmetrics import Metric

import neuralcompression.functional as ncF
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
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Optional[Callable] = None,
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
            dist_sync_on_step: see ``torchmetrics.Metric`` documentation.
            process_group: see ``torchmetrics.Metric`` documentation.
            dist_sync_fn: see ``torchmetrics.Metric`` documentation.
        """

        super().__init__(
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
        self.score_sum += ncF.multiscale_structural_similarity(
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
