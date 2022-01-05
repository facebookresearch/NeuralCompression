# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy
import pytest
import tensorflow
import tensorflow_addons
import torch
from utils import create_input

from neuralcompression.functional import dense_image_warp


@pytest.mark.parametrize(
    "shape, seed",
    [
        ((3, 3, 72, 64), 0),
        ((5, 3, 55, 18), 1),
        ((6, 3, 73, 35), 2),
    ],
)
def test_dense_image_warp(shape, seed):
    rng = numpy.random.default_rng(seed)
    image = create_input(shape)

    # this test won't work for completely random warps - about a 10% error
    # so we create a single vector flow and rotate it randomly
    flow = (
        numpy.stack(
            (
                numpy.ones(shape[0] * shape[-2] * shape[-1]),
                numpy.zeros(shape[0] * shape[-2] * shape[-1]),
            )
        )
        * (rng.uniform() / 10)
    )

    angle = rng.uniform() * 2 * numpy.pi
    cos_val, sin_val = numpy.cos(angle), numpy.sin(angle)
    rot_mat = numpy.array(((cos_val, -sin_val), (sin_val, cos_val)))
    flow = numpy.reshape(
        numpy.transpose(rot_mat @ flow), (shape[0], shape[-2], shape[-1], 2)
    )
    flow = torch.tensor(flow).to(image)

    image_warp = dense_image_warp(image, flow, align_corners=True)

    # tensorflow version operates on pixel indices rather than (-1, 1) range
    tf_image = tensorflow.convert_to_tensor(image.permute(0, 2, 3, 1).numpy())
    tf_flow = tensorflow.convert_to_tensor(
        numpy.flip(flow.numpy(), -1) * ((numpy.array(shape[-2:]) - 1) / 2)
    )
    tf_image_warp = tensorflow_addons.image.dense_image_warp(tf_image, -tf_flow)

    tf_image_warp = torch.tensor(tf_image_warp.numpy()).permute(0, 3, 1, 2)

    assert torch.allclose(tf_image_warp, image_warp)
