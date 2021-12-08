# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import numpy as np
import pytest
import tensorflow as tf
import tensorflow_addons as tfa
import torch

import _hsv2rgb
import _optical_flow_to_color
from utils import create_input

import neuralcompression.functional as ncF


@pytest.mark.parametrize(
    "shape, seed", [((5, 3, 64, 64), 0), ((2, 3, 76, 55), 1), ((10, 3, 32, 64), 2)]
)
def test_hsv2rgb(shape, seed):
    # verify correctness of hsv2rgb function by comparing to OpenCV impl.
    rng = np.random.default_rng(seed)
    hsvs = rng.uniform(low=0.0, high=1.0, size=shape)
    hsvs[:, 0] *= 360.0
    hsvs = torch.tensor(hsvs, dtype=torch.float32)

    rgb_torch = _hsv2rgb.hsv2rgb(hsvs)

    for i, hsv in enumerate(hsvs):
        bgr = np.flip(cv2.cvtColor(hsv.permute(1, 2, 0).numpy(), cv2.COLOR_HSV2BGR), 2)

        # low atol due to float32
        assert np.allclose(bgr, rgb_torch[i].permute(1, 2, 0).numpy(), atol=1e-6)


@pytest.mark.parametrize(
    "shape, seed",
    [
        ((3, 72, 64, 5), 0),
        ((5, 55, 18, 2), 1),
        ((6, 73, 35, 10), 2),
    ],
)
def test_information_content(shape, seed):
    # test all reductions of coding cost
    # also check base-2 and base-10
    rng = np.random.default_rng(seed)
    probabilities = torch.tensor(rng.uniform(size=shape))

    def batch_el_reduction(x):
        return torch.stack([torch.sum(el) for el in x])

    base_ops = {2: torch.log2, 10: torch.log10}
    reduction_ops = {
        "sum": torch.sum,
        "batch_el": batch_el_reduction,
        "none": lambda x: x,
    }

    for base in (2, 10):
        for reduction in ("sum", "batch_el", "none"):
            torch_cost = -1 * reduction_ops[reduction](base_ops[base](probabilities))
            assert torch.allclose(
                ncF.information_content(probabilities, reduction=reduction, base=base),
                torch_cost,
            )


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
    rgb_flow = _optical_flow_to_color.optical_flow_to_color(flow)

    assert tuple(rgb_flow.shape) == (shape[0], 3, shape[2], shape[3])
