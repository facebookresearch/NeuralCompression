# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy
import pytest
import torch

import neuralcompression.functional as ncF

from utils import create_input


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
        rng: numpy.random.Generator = numpy.random.default_rng(seed)
        flow = torch.tensor(rng.normal(size=shape))

    flow = flow / flow.abs().max()  # max abs value must be 1
    rgb_flow = ncF.optical_flow_to_color(flow)

    assert tuple(rgb_flow.shape) == (shape[0], 3, shape[2], shape[3])
