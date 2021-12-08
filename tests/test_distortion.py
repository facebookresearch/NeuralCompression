# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import tensorflow as tf
import numpy as np
import torch

from neuralcompression.functional import (
    learned_perceptual_image_patch_similarity,
)
from functional import multiscale_structural_similarity
from neuralcompression.metrics import (
    LearnedPerceptualImagePatchSimilarity,
    MultiscaleStructuralSimilarity,
)


def tf_ms_ssim(x, y):
    # Converting NCHW image format to NHWC
    x = tf.convert_to_tensor(x.permute(0, 2, 3, 1))
    y = tf.convert_to_tensor(y.permute(0, 2, 3, 1))
    return torch.tensor(tf.image.ssim_multiscale(x, y, max_val=1).numpy())


def rand_im(shape, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return torch.tensor(rng.uniform(size=shape), dtype=torch.get_default_dtype())


@pytest.mark.parametrize(
    "num_channels, img_size, batch_size, seed",
    [(1, 256, 1, 100), (1, 250, 4, 101), (10, 256, 1, 102)],
)
def test_ms_ssim_functional(num_channels, img_size, batch_size, seed):
    # Testing basic properties of the functional MS-SSIM API,
    # plus comparing its output to Tensorflow's implementation.
    rng = np.random.default_rng(seed)

    img1 = rand_im([batch_size, num_channels, img_size, img_size], rng)
    img2 = rand_im([batch_size, num_channels, img_size, img_size], rng)

    assert torch.allclose(
        multiscale_structural_similarity(img1, img1), torch.tensor(1.0)
    )

    assert multiscale_structural_similarity(
        img1, img1, reduction="none"
    ).shape == torch.Size([batch_size])
    assert torch.allclose(
        multiscale_structural_similarity(img1, img1, reduction="sum"),
        torch.tensor(batch_size).float(),
    )
    assert torch.allclose(
        multiscale_structural_similarity(img1, img1, reduction="mean"),
        torch.tensor(1.0),
    )

    score = multiscale_structural_similarity(img1, img2, reduction="none")
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

    rng = np.random.default_rng(seed)
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
        assert torch.allclose(score, multiscale_structural_similarity(img1, img2))

    assert torch.allclose(
        metric.compute(),
        multiscale_structural_similarity(
            torch.cat(imgs1, dim=0), torch.cat(imgs2, dim=0)
        ),
    )


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

    rng = np.random.default_rng(seed)
    metric = LearnedPerceptualImagePatchSimilarity()

    imgs1 = [rand_im([batch_size, 3, *input_shape], rng) for batch_size in batch_sizes]
    imgs2 = [rand_im([batch_size, 3, *input_shape], rng) for batch_size in batch_sizes]

    for img1, img2 in zip(imgs1, imgs2):
        score = metric(img1, img2)
        assert torch.allclose(
            score, learned_perceptual_image_patch_similarity(img1, img2)
        )

    assert torch.allclose(
        metric.compute(),
        learned_perceptual_image_patch_similarity(
            torch.cat(imgs1, dim=0), torch.cat(imgs2, dim=0)
        ),
    )
