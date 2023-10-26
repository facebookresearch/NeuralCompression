# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from conftest import MockBackbone
from torch import Tensor

import neuralcompression.functional as ncF
import neuralcompression.metrics._fid
from neuralcompression.metrics import FrechetInceptionDistance


@pytest.mark.parametrize("num_samples", [5])
def test_fid(num_samples: int, arange_4d_image: Tensor, monkeypatch):
    if arange_4d_image.shape[1] != 3:
        return

    rng = torch.Generator()
    rng.manual_seed(55)

    monkeypatch.setattr(
        neuralcompression.metrics._fid, "NoTrainInceptionV3", MockBackbone
    )
    metric = FrechetInceptionDistance()

    for _ in range(num_samples):
        sample1 = torch.randn(size=arange_4d_image.shape, generator=rng).abs()
        sample1 = sample1 / sample1.max()
        sample1 = ncF.image_to_255_scale(sample1, torch.uint8)
        sample2 = torch.randn(size=arange_4d_image.shape, generator=rng).abs()
        sample2 = sample2 / sample2.max()
        sample2 = ncF.image_to_255_scale(sample2, torch.uint8)

        metric.update(sample1, True)
        metric.update(sample2, False)

    metric.compute()
