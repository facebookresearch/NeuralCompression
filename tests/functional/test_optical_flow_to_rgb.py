# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_addons as tfa
import torch
from utils import create_input

import neuralcompression.functional as ncF
from neuralcompression.functional import optical_flow_to_color


@pytest.mark.parametrize(
    "shape, seed",
    [
        ((3, 3, 72, 64), 0),
        ((5, 3, 55, 18), 1),
        ((6, 3, 73, 35), 2),
    ],
)
def test_dense_image_warp(shape, seed):
    rng = np.random.default_rng(seed)
    image = create_input(shape)

    # this test won't work for completely random warps - about a 10% error
    # so we create a single vector flow and rotate it randomly
    flow = (
        np.stack(
            (
                np.ones(shape[0] * shape[-2] * shape[-1]),
                np.zeros(shape[0] * shape[-2] * shape[-1]),
            )
        )
        * (rng.uniform() / 10)
    )
    angle = rng.uniform() * 2 * np.pi
    cos_val, sin_val = np.cos(angle), np.sin(angle)
    rot_mat = np.array(((cos_val, -sin_val), (sin_val, cos_val)))
    flow = np.reshape(np.transpose(rot_mat @ flow), (shape[0], shape[-2], shape[-1], 2))
    flow = torch.tensor(flow).to(image)

    image_warp = ncF.dense_image_warp(image, flow, align_corners=True)

    # tensorflow version operates on pixel indices rather than (-1, 1) range
    tf_image = tf.convert_to_tensor(image.permute(0, 2, 3, 1).numpy())
    tf_flow = tf.convert_to_tensor(
        np.flip(flow.numpy(), -1) * ((np.array(shape[-2:]) - 1) / 2)
    )
    tf_image_warp = tfa.image.dense_image_warp(tf_image, -tf_flow)

    tf_image_warp = torch.tensor(tf_image_warp.numpy()).permute(0, 3, 1, 2)

    assert torch.allclose(tf_image_warp, image_warp)


@pytest.mark.parametrize(
    "shape, seed",
    [
        ((5, 2, 64, 64), 3),
        ((2, 2, 76, 55), 4),
        ((10, 2, 32, 64), 5),
        ((7, 2, 100, 70), None),
    ],
)
def test_optical_flow_to_rgb(shape, seed):
    # a simple run-without-exception
    # HSV color correctness verified in test_hsv2rgb
    if seed is None:
        flow = create_input(shape)
    else:
        rng: np.random.Generator = np.random.default_rng(seed)
        flow = torch.tensor(rng.normal(size=shape))

    flow = flow / flow.abs().max()  # max abs value must be 1
    rgb_flow = optical_flow_to_color(flow)

    assert tuple(rgb_flow.shape) == (shape[0], 3, shape[2], shape[3])
