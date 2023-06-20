# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch_fidelity.metric_kid import kid_features_to_metric
from torchmetrics.image.kid import NoTrainInceptionV3
from torchmetrics.metric import Metric


class KernelInceptionDistance(Metric):
    """
    Kernel Inception Distance.

    This is a minimalist torchmetrics wrapper for the KID calculation in
    torch-fidelity. Unlike the torchmetrics implementation, intermediate
    features are stored on CPU prior to final metric calculation.

    Args:
        feature: An integer that indicates the inceptionv3 feature layer to
            choose. Can be one of the following: 64, 192, 768, 2048.
        subsets: Number of subsets to calculate the mean and standard
            deviation scores over
        subset_size: Number of randomly picked samples in each subset
        degree: Degree of the polynomial kernel function
        gamma: Scale-length of polynomial kernel. If set to ``None`` will be
            automatically set to the feature size
        coef: Bias term in the polynomial kernel.
        reset_real_features: Whether to also reset the real features. Since in
            many cases the real dataset does not change, the features can
            cached them to avoid recomputing them which is costly. Set this to
            ``False`` if your dataset does not change.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more
                info.
    """

    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False

    real_features: List[Tensor]
    fake_features: List[Tensor]

    def __init__(
        self,
        feature: Union[str, int, Module] = 2048,
        subsets: int = 100,
        subset_size: int = 1000,
        degree: int = 3,
        gamma: Optional[float] = None,
        coef: float = 1.0,
        normalize: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.inception: Module = NoTrainInceptionV3(
            name="inception-v3-compat", features_list=[str(feature)]
        )
        self.subsets = subsets
        self.subset_size = subset_size
        self.degree = degree
        self.gamma = gamma
        self.coef = coef

        self.normalize = normalize

        # states for extracted features
        self.add_state("real_features", [], dist_reduce_fx=None)
        self.add_state("fake_features", [], dist_reduce_fx=None)

    def update(self, imgs: Tensor, real: bool) -> None:
        imgs = (imgs * 255).byte() if self.normalize else imgs
        features = self.inception(imgs)

        if real:
            self.real_features.append(features.cpu())
        else:
            self.fake_features.append(features.cpu())

    def compute(self) -> Tuple[Tensor, Tensor]:
        output = kid_features_to_metric(
            torch.cat(self.real_features).to(torch.float64),
            torch.cat(self.fake_features).to(torch.float64),
        )
        return (
            torch.tensor(output["kernel_inception_distance_mean"]),
            torch.tensor(output["kernel_inception_distance_std"]),
        )
