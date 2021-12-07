# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from neuralcompression.models import ScaleHyperprior


@pytest.mark.parametrize(
    "network_channels,compression_channels,img_size", [(128, 256, 256), (256, 128, 100)]
)
def test_hyperprior_forward(network_channels, compression_channels, img_size):
    # Tests the scale hyperprior model's forward function.
    #
    # Across different model and input image sizes, this tests verifies
    # that the scale hyperprior model's forward function doesn't crash.
    # This forward function runs an image through the model's encoder and
    # decoder networks, using a differentiable stand-in for quantization.
    # This test also verifies that the model's image reconstruction has
    # the same shape as the original image.

    inp = torch.randn(1, 3, img_size, img_size)
    model = ScaleHyperprior(
        network_channels=network_channels, compression_channels=compression_channels
    )
    outputs = model(inp)
    assert outputs[0].shape == inp.shape


@pytest.mark.parametrize(
    "network_channels,compression_channels,img_size", [(128, 256, 256), (256, 128, 100)]
)
def test_hyperprior_compressison(network_channels, compression_channels, img_size):
    # Exercizes the scale hyperprior compress and decompress functions.
    #
    # Tests that the model can successfully compress and decompress
    # an image without crashing, and that the decompressed reconstruction
    # is the same shape as the original input.

    inp = torch.randn(1, 3, img_size, img_size)
    model = ScaleHyperprior(
        network_channels=network_channels, compression_channels=compression_channels
    )
    updated = model.update()
    assert updated, "Model did not update even though no update previously occurred"

    compression_outputs = model.compress(inp)
    reconstruction = model.decompress(*compression_outputs)
    assert reconstruction.shape == inp.shape
