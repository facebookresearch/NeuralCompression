# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import torch
from utils import create_input

import neuralcompression.functional as ncF
from neuralcompression.models.deep_video_compression import (
    DVC,
    DVCCompressionDecoder,
    DVCCompressionEncoder,
    DVCMotionCompensationModel,
    DVCPyramidFlowEstimator,
)


def gen_random_image(shape, rng):
    return torch.tensor(rng.uniform(size=shape), dtype=torch.get_default_dtype())


@pytest.mark.parametrize(
    "shape",
    [([5, 2, 180, 160]), ([3, 2, 96, 64]), ([1, 2, 32, 33])],
)
def test_dvc_compression_autoencoder(shape):
    # test the compression autoencoder via run-without-exception
    x = create_input(shape)

    for use_gdn in (True, False):
        encoder = DVCCompressionEncoder(use_gdn=use_gdn)
        decoder = DVCCompressionDecoder(use_gdn=use_gdn)

        # test basic forward runs without exception
        encoded, sizes = encoder(x)
        decoded = decoder(encoded, sizes)

        assert list(decoded.shape) == shape


@pytest.mark.parametrize(
    "shape",
    [([5, 3, 160, 160]), ([3, 3, 64, 64]), ([1, 3, 32, 32])],
)
def test_dvc_motion_compensation(shape):
    # test the individual motion compensation block via run-without-exception
    rng = np.random.default_rng(123)
    image1 = gen_random_image(shape, rng)
    new_shape = shape[:]  # shallow copy
    new_shape[1] = 2
    flow = gen_random_image(new_shape, rng)

    # test basic forward runs without exception
    layer = DVCMotionCompensationModel()
    output = layer(ncF.dense_image_warp(image1, flow.permute(0, 2, 3, 1)), image1, flow)

    assert list(output.shape) == shape


@pytest.mark.parametrize(
    "shape",
    [(5, 3, 160, 160), (3, 3, 64, 64), (1, 3, 32, 32)],
)
def test_dvc_integrated_basic(shape):
    # test integrated, fully-composed DVC model via run-without-exception
    rng = np.random.default_rng(1234)
    image1 = gen_random_image(shape, rng)
    image2 = gen_random_image(shape, rng)

    # test basic forward runs without exception
    model = DVC()
    output = model(image1, image2)

    assert output.residual.shape == image2.shape
    assert output.image2_est.shape == image2.shape
    assert tuple(output.flow.shape) == (
        image1.shape[0],
        2,
        image1.shape[2],
        image1.shape[3],
    )


@pytest.mark.parametrize(
    "shape",
    [(1, 3, 32, 32), (2, 3, 32, 32)],
)
def test_dvc_integrated_compress(shape):
    # test the overall integrated compression pipeline
    # (to be used by final trained model)
    rng = np.random.default_rng(1234)
    image1 = gen_random_image(shape, rng)
    image2 = gen_random_image(shape, rng)

    # test compression functions
    model = DVC()
    model.update(force=True)
    model = model.eval()
    compressed = model.compress(image1, image2)
    image2_est = model.decompress(image1, compressed)

    assert image2_est.shape == image2.shape


@pytest.mark.parametrize(
    "shape",
    [([5, 3, 160, 160]), ([3, 3, 64, 64]), ([1, 3, 32, 32])],
)
def test_dvc_pyramid_flow_estimator(shape):
    # checks output shapes of training mode for pyramidal flow
    rng = np.random.default_rng(12345)
    image1 = gen_random_image(shape, rng)
    image2 = gen_random_image(shape, rng)

    # test basic forward runs without exception
    layer = DVCPyramidFlowEstimator()
    flow, outputs = layer.calculate_flow_with_image_pairs(image1, image2)

    output_shapes = [(shape[0], 3, shape[2], shape[3])]
    for _ in range(1, len(layer.model_levels)):
        output_shapes.append(
            (shape[0], 3, output_shapes[-1][2] // 2, output_shapes[-1][2] // 2)
        )

    for output in outputs:
        output_shape = output_shapes.pop()
        assert tuple(output[0].shape) == output_shape
        assert tuple(output[1].shape) == output_shape

    assert tuple(flow.shape) == (image1.shape[0], 2, image1.shape[2], image1.shape[3])
