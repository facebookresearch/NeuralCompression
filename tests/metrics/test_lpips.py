# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy
import pytest
import torch

import neuralcompression.functional as ncF
from neuralcompression.metrics import LearnedPerceptualImagePatchSimilarity
from utils import rand_im


@pytest.mark.parametrize(
    "batch_sizes, input_shape, seed",
    [
        [[3, 2, 5], (64, 64), 0],
        [[1], (35, 35), 1],
        [[1, 2], (129, 31), 2],
        [[2, 1], (50, 123), 3],
    ],
)
def test_lpips(batch_sizes, input_shape, seed):
    # Tests that the LPIPS module and functional APIs
    # produce identical results across a variety of
    # image and batch sizes.

    rng = numpy.random.default_rng(seed)
    metric = LearnedPerceptualImagePatchSimilarity()

    imgs1 = [rand_im([batch_size, 3, *input_shape], rng) for batch_size in batch_sizes]
    imgs2 = [rand_im([batch_size, 3, *input_shape], rng) for batch_size in batch_sizes]

    for img1, img2 in zip(imgs1, imgs2):
        score = metric(img1, img2)
        assert torch.allclose(score, ncF.lpips(img1, img2))

    assert torch.allclose(
        metric.compute(),
        ncF.lpips(torch.cat(imgs1, dim=0), torch.cat(imgs2, dim=0)),
    )
