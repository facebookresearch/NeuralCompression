# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import numpy
import torch

import neuralcompression.functional as ncF
from neuralcompression.metrics import MultiscaleStructuralSimilarity
from utils import tf_ms_ssim, rand_im


@pytest.mark.parametrize(
    "num_channels, img_size, batch_size, seed",
    [(1, 256, 1, 100), (1, 250, 4, 101), (10, 256, 1, 102)],
)
def test_ms_ssim_functional(num_channels, img_size, batch_size, seed):
    # Testing basic properties of the functional MS-SSIM API,
    # plus comparing its output to Tensorflow's implementation.
    rng = numpy.random.default_rng(seed)

    img1 = rand_im([batch_size, num_channels, img_size, img_size], rng)
    img2 = rand_im([batch_size, num_channels, img_size, img_size], rng)

    assert torch.allclose(ncF.ms_ssim(img1, img1), torch.tensor(1.0))

    assert ncF.ms_ssim(img1, img1, reduction="none").shape == torch.Size([batch_size])
    assert torch.allclose(
        ncF.ms_ssim(img1, img1, reduction="sum"),
        torch.tensor(batch_size).float(),
    )
    assert torch.allclose(
        ncF.ms_ssim(img1, img1, reduction="mean"),
        torch.tensor(1.0),
    )

    score = ncF.ms_ssim(img1, img2, reduction="none")
    reference = tf_ms_ssim(img1, img2)
    assert torch.allclose(score, reference, atol=1e-5)


@pytest.mark.parametrize(
    "num_channels, img_size, batch_sizes, seed",
    [(2, 256, [1, 5, 3], 123)],
)
def test_ms_ssim_module(num_channels, img_size, batch_sizes, seed):
    # Tests that the MS-SSIM Module API produces identical
    # results to the functional API after accumulating
    # variable-sized batches.

    rng = numpy.random.default_rng(seed)
    metric = MultiscaleStructuralSimilarity()

    imgs1 = [
        rand_im([batch_size, num_channels, img_size, img_size], rng)
        for batch_size in batch_sizes
    ]
    imgs2 = [
        rand_im([batch_size, num_channels, img_size, img_size], rng)
        for batch_size in batch_sizes
    ]

    for img1, img2 in zip(imgs1, imgs2):
        score = metric(img1, img2)
        assert torch.allclose(score, ncF.ms_ssim(img1, img2))

    assert torch.allclose(
        metric.compute(),
        ncF.ms_ssim(torch.cat(imgs1, dim=0), torch.cat(imgs2, dim=0)),
    )
