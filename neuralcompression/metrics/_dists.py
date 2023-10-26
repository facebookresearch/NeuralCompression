# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List

import torch
from DISTS_pytorch import DISTS as _DISTS
from torch import Tensor
from torchmetrics.metric import Metric
from typing_extensions import Literal


class NoTrainDists(_DISTS):
    def train(self, mode: bool = True) -> "NoTrainDists":
        """keep network in evaluation mode."""
        return super().train(False)


def _valid_img(img: Tensor) -> bool:
    """check that input is a valid image to the network."""
    value_check = bool((img.max() <= 1.0) and (img.min() >= 0.0))
    return (img.ndim == 4) and (img.shape[1] == 3) and value_check


class DeepImageStructureTextureSimilarity(Metric):
    """
    DISTS PyTorch Metric.

    This torchmetric class implements the metric described in the following
    paper:

    Image Quality Assessment: Unifying Structure and Texture Similarity
    Keyan Ding, Kede Ma, Shiqi Wang, Eero P. Simoncelli

    It has similar properties to LPIPS, but with a greater emphasis on texture.

    Args:
        reduction: str indicating how to reduce over the batch dimension.
            Choose between `'sum'` or `'mean'`.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more
            info.
    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False

    real_features: List[Tensor]
    fake_features: List[Tensor]

    sum_scores: Tensor
    total: Tensor

    # due to the use of named tuple in the backbone the net variable cannot be
    # scripted
    __jit_ignored_attributes__ = ["net"]

    def __init__(
        self, reduction: Literal["sum", "mean"] = "mean", **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.net = NoTrainDists()

        valid_reduction = ("mean", "sum")
        if reduction not in valid_reduction:
            raise ValueError(
                f"Argument `reduction` must be one of {valid_reduction}, "
                f"but got {reduction}"
            )
        self.reduction = reduction

        self.add_state("sum_scores", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, img1: Tensor, img2: Tensor) -> None:  # type: ignore
        """Update internal states with lpips score.
        Args:
            img1: tensor with images of shape ``[N, 3, H, W]``
            img2: tensor with images of shape ``[N, 3, H, W]``
        """
        if not (_valid_img(img1) and _valid_img(img2)):
            raise ValueError(
                "Expected both input arguments to be normalized tensors with "
                f"shape [N, 3, H, W]. Got input with shape {img1.shape} and "
                f"{img2.shape} and values in range {[img1.min(), img1.max()]} "
                f"and {[img2.min(), img2.max()]} when all values are expected "
                f"to be in the {[0,1]} range."
            )
        loss = self.net(img1, img2).squeeze()
        self.sum_scores += loss.sum()
        self.total += img1.shape[0]

    def compute(self) -> Tensor:
        """Compute final perceptual similarity metric."""
        if self.reduction == "mean":
            return self.sum_scores / self.total
        if self.reduction == "sum":
            return self.sum_scores
