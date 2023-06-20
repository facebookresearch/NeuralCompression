# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch import Tensor

import neuralcompression.functional as ncF
from neuralcompression.metrics import KernelInceptionDistance


@pytest.mark.parametrize("num_samples", [5])
def test_dists(num_samples: int, arange_4d_image: Tensor):
    if arange_4d_image.shape[1] != 3:
        return

    metric = KernelInceptionDistance()

    for _ in range(num_samples):
        sample1 = torch.randn_like(arange_4d_image).abs()
        sample1 = sample1 / sample1.max()
        sample1 = ncF.image_to_255_scale(sample1, torch.uint8)
        sample2 = torch.randn_like(arange_4d_image).abs()
        sample2 = sample2 / sample2.max()
        sample2 = ncF.image_to_255_scale(sample2, torch.uint8)

        metric.update(sample1, True)
        metric.update(sample2, False)

    # running compute() requires too many samples
