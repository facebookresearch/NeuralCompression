# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from torch import Tensor

from neuralcompression.models._hific_encoder_decoder import HiFiCEncoder


@pytest.mark.parametrize("latent_features", [220, 110, 80])
def test_hific_encoder_forward(latent_features: int, arange_4d_image: Tensor):
    model = HiFiCEncoder(
        in_channels=arange_4d_image.shape[1], latent_features=latent_features
    )

    output = model(arange_4d_image)

    output_shape = (
        arange_4d_image.shape[0],
        latent_features,
        arange_4d_image.shape[2] // 16,
        arange_4d_image.shape[3] // 16,
    )

    assert output.shape == output_shape
