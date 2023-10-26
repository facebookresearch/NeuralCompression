# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch import Tensor

import neuralcompression.metrics._fid
import neuralcompression.metrics._fid_swav
import neuralcompression.metrics._kid
from neuralcompression.metrics import (
    FrechetInceptionDistance,
    FrechetInceptionDistanceSwAV,
    KernelInceptionDistance,
    update_patch_fid,
)


@pytest.mark.parametrize("num_samples", [5])
def test_dists(num_samples: int, arange_4d_image: Tensor, monkeypatch, mock_backbone):
    if arange_4d_image.shape[1] != 3:
        return

    monkeypatch.setattr(
        neuralcompression.metrics._fid, "NoTrainInceptionV3", mock_backbone
    )
    monkeypatch.setattr(
        neuralcompression.metrics._fid_swav, "NoTrainSwAV", mock_backbone
    )
    monkeypatch.setattr(
        neuralcompression.metrics._kid, "NoTrainInceptionV3", mock_backbone
    )

    fid_metric = FrechetInceptionDistance()
    fid_swav_metric = FrechetInceptionDistanceSwAV()
    kid_metric = KernelInceptionDistance()

    for _ in range(num_samples):
        sample1 = torch.randn_like(arange_4d_image).abs()
        sample1 = sample1 / sample1.max()
        sample2 = torch.randn_like(arange_4d_image).abs()
        sample2 = sample2 / sample2.max()

        update_patch_fid(
            sample1, sample2, fid_metric, fid_swav_metric, kid_metric, patch_size=128
        )
