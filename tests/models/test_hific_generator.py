# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch import Generator, Tensor

from neuralcompression.models._hific_encoder_decoder import HiFiCGenerator


@pytest.mark.parametrize("latent_features", [220, 110, 80])
def test_hific_generator_forward(latent_features: int, arange_4d_image: Tensor):
    rng = Generator()
    latent_shape = (
        arange_4d_image.shape[0],
        latent_features,
        arange_4d_image.shape[2] // 16,
        arange_4d_image.shape[3] // 16,
    )
    latent = torch.randn(latent_shape, generator=rng)
    model = HiFiCGenerator(
        image_channels=arange_4d_image.shape[1], latent_features=latent_features
    )

    output = model(latent)

    assert output.shape == arange_4d_image.shape
