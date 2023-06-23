# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch import Tensor

from neuralcompression.metrics import MultiscaleStructuralSimilarity


@pytest.mark.parametrize("num_samples", [5])
def test_msssim(num_samples: int, arange_4d_image: Tensor):
    if arange_4d_image.shape[2] < 128 or arange_4d_image.shape[3] < 128:
        return

    metric = MultiscaleStructuralSimilarity()

    for _ in range(num_samples):
        sample1 = torch.randn_like(arange_4d_image).abs()
        sample1 = sample1 / sample1.max()
        sample2 = torch.randn_like(arange_4d_image).abs()
        sample2 = sample2 / sample2.max()

        metric.update(sample1, sample2)

    metric.compute()
