# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle

import pytest
import torch

from neuralcompression.functional import dense_image_warp


@pytest.mark.timeout(method="thread")
def test_dense_image_warp():
    with open("tests/cached_data/dense_image_warp.pkl", "rb") as f:
        data = pickle.load(f)

    for sample in data:
        image = sample["image"]
        flow = sample["flow"]
        tf_image_warp = sample["tf_image_warp"]

        image_warp = dense_image_warp(image, flow, align_corners=True)

        assert torch.allclose(tf_image_warp, image_warp)
