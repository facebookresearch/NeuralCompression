"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pytest
import tensorflow as tf
import numpy as np
import torch

from neuralcompression.functional import (
    learned_perceptual_image_patch_similarity,
    multiscale_structural_similarity,
)
from neuralcompression.metrics import (
    LearnedPerceptualImagePatchSimilarity,
    MultiscaleStructuralSimilarity,
)


def tf_ms_ssim(x, y):
    # Converting NCHW image format to NHWC
    x = tf.convert_to_tensor(x.permute(0, 2, 3, 1))
    y = tf.convert_to_tensor(y.permute(0, 2, 3, 1))
    return torch.tensor(tf.image.ssim_multiscale(x, y, max_val=1).numpy())


def rand_im(shape, seed=12345):
    rng = np.random.default_rng(seed)

    return torch.tensor(rng.uniform(size=shape), dtype=torch.get_default_dtype())


@pytest.mark.parametrize(
    "num_channels,img_size,batch_size",
    [(1, 256, 1), (1, 250, 4), (10, 256, 1)],
)
def test_ms_ssim_functional(num_channels, img_size, batch_size):
    # Testing basic properties of the functional MS-SSIM API,
    # plus comparing its output to Tensorflow's implementation.

    img1 = rand_im([batch_size, num_channels, img_size, img_size], 100)
    img2 = rand_im([batch_size, num_channels, img_size, img_size], 101)

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
    "num_channels,img_size,batch_sizes",
    [(2, 256, [1, 5, 3])],
)
def test_ms_ssim_module(num_channels, img_size, batch_sizes):
    # Tests that the MS-SSIM Module API produces identical
    # results to the functional API after accumulating
    # variable-sized batches.

    metric = MultiscaleStructuralSimilarity()

    imgs1 = [
        rand_im([batch_size, num_channels, img_size, img_size], 123)
        for batch_size in batch_sizes
    ]
    imgs2 = [
        rand_im([batch_size, num_channels, img_size, img_size], 124)
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
    "batch_sizes, input_shape",
    [
        [[3, 2, 5], (64, 64)],
        [[1], (35, 35)],
        [[1, 2], (129, 31)],
        [[2, 1], (50, 123)],
    ],
)
def test_lpips(batch_sizes, input_shape):
    # Tests that the LPIPS module and functional APIs
    # produce identical results across a variety of
    # image and batch sizes.

    metric = LearnedPerceptualImagePatchSimilarity()

    imgs1 = [rand_im([batch_size, 3, *input_shape], 0) for batch_size in batch_sizes]
    imgs2 = [rand_im([batch_size, 3, *input_shape], 1) for batch_size in batch_sizes]

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
