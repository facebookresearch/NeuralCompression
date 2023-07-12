# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from torch_fidelity.metric_fid import (
    fid_features_to_statistics,
    fid_statistics_to_metric,
)
from torchmetrics.metric import Metric
from torchvision.transforms import Normalize


class NoTrainSwAV(Module):
    def __init__(self, torch_hub: Optional[str] = None) -> None:
        super().__init__()
        if torch_hub is None:
            torch_hub = "facebookresearch/swav:main"

        self.model = torch.hub.load(torch_hub, "resnet50")
        self.model.fc = nn.Identity()
        # put into evaluation mode
        self.eval()

    def train(self, mode: bool = True) -> "NoTrainSwAV":
        """keep the model in eval mode."""
        return super().train(False)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


def _compute_fid_from_features(
    features_real: Tensor,
    features_fake: Tensor,
) -> Tensor:
    stat_1 = fid_features_to_statistics(features_real)
    stat_2 = fid_features_to_statistics(features_fake)

    metric = fid_statistics_to_metric(stat_1, stat_2, False)[
        "frechet_inception_distance"
    ]

    return torch.tensor(metric)


class FrechetInceptionDistanceSwAV(Metric):
    """
    FID with SwAV backbone.

    This implements the Frechet 'Inception' Distance metric, but with a SwAV
    ResNet backbone instead of an Inception V3 backbone. The details are
    described in the following paper:

    The Role of ImageNet Classes in Fréchet Inception Distance
    Tuomas Kynkäänniemi, Tero Karras, Miika Aittala, Timo Aila, Jaakko Lehtinen

    Args:
        normalize: Whether to apply ImageNet normalization prior to running the
            SwAV ResNet.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more
            info.
    """

    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False

    real_features: List[Tensor]
    fake_features: List[Tensor]

    def __init__(self, normalize: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.feature_extractor = NoTrainSwAV()

        if not isinstance(normalize, bool):
            raise ValueError("Argument `normalize` expected to be a bool")

        if normalize:
            self.normalize = Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]
            )
        else:
            self.normalize = nn.Identity()

        self.add_state("real_features", [], dist_reduce_fx="cat")
        self.add_state("fake_features", [], dist_reduce_fx="cat")

    def update(self, imgs: Tensor, real: bool) -> None:  # type: ignore
        imgs = self.normalize(imgs.float() / 255.0)

        features = self.feature_extractor(imgs)

        if features.dim() == 1:
            features = features.unsqueeze(0)
        if real:
            self.real_features.append(features.cpu())
        else:
            self.fake_features.append(features.cpu())

    def compute(self) -> Tensor:
        return _compute_fid_from_features(
            torch.cat(self.real_features).to(torch.float64),
            torch.cat(self.fake_features).to(torch.float64),
        )
