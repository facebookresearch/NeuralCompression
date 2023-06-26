# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn as nn

from neuralcompression.models import ConditionalUnetDiscriminator, UnetDiscriminator


def unet_image(shape):
    rng = torch.Generator()
    rng.manual_seed(int(torch.prod(torch.tensor(shape))))

    image = torch.randn(size=shape, generator=rng)

    return image


@pytest.mark.parametrize("shape", [(5, 3, 128, 128)])
@pytest.mark.parametrize("norm_type", ["identity", "spectral", "instance"])
@pytest.mark.parametrize(
    "output_channels,downsampling_factor", [(5, 1), (1025, 8), (257, 16)]
)
def test_unet_discriminator(shape, norm_type, output_channels, downsampling_factor):
    image = unet_image(shape)
    model = UnetDiscriminator(
        output_downsampling_factor=downsampling_factor,
        output_channels=output_channels,
        norm_type=norm_type,
    )

    expected_shape = (
        shape[0],
        output_channels,
        shape[2] // downsampling_factor,
        shape[3] // downsampling_factor,
    )

    if norm_type == "identity":
        test_model = model.model[0]

        while not isinstance(test_model, nn.Identity):
            assert isinstance(test_model.left_block[0], nn.Identity)
            assert isinstance(test_model.left_block[3], nn.Identity)

            test_model = test_model.child
    if norm_type == "spectral":
        test_model = model.model[0]

        while not isinstance(test_model, nn.Identity):
            assert nn.utils.parametrize.is_parametrized(test_model.left_block[2])
            assert nn.utils.parametrize.is_parametrized(test_model.left_block[5])

            test_model = test_model.child
    if norm_type == "instance":
        test_model = model.model[0]
        top = True

        while not isinstance(test_model, nn.Identity):
            if top:
                top = False
            else:
                assert isinstance(test_model.left_block[0], nn.InstanceNorm2d)
            assert isinstance(test_model.left_block[3], nn.InstanceNorm2d)

            test_model = test_model.child

    output = model(image)

    assert output.shape == expected_shape


@pytest.mark.parametrize("shape", [(5, 3, 128, 128)])
@pytest.mark.parametrize("context_shape", [(5, 320, 4, 4)])
@pytest.mark.parametrize("norm_type", ["identity"])
@pytest.mark.parametrize("output_channels,downsampling_factor", [(1025, 8)])
def test_condunet_discriminator(
    shape, context_shape, norm_type, output_channels, downsampling_factor
):
    image = unet_image(shape)
    context = unet_image(context_shape)
    model = ConditionalUnetDiscriminator(
        output_downsampling_factor=downsampling_factor,
        output_channels=output_channels,
        context_in=context.shape[1],
        norm_type=norm_type,
    )

    expected_shape = (
        shape[0],
        output_channels,
        shape[2] // downsampling_factor,
        shape[3] // downsampling_factor,
    )

    output = model(image, context)

    assert output.shape == expected_shape
