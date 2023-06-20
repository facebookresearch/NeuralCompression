# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch import Tensor

from neuralcompression.models import HiFiCAutoencoder


@pytest.mark.parametrize(
    "latent_features,hyper_features,num_residual_blocks",
    [(220, 320, 9), (110, 160, 5), (80, 40, 3)],
)
def test_hific_encoder_forward(
    latent_features: int,
    hyper_features: int,
    num_residual_blocks: int,
    arange_4d_image: Tensor,
):
    model = HiFiCAutoencoder(
        in_channels=arange_4d_image.shape[1],
        latent_features=latent_features,
        hyper_features=hyper_features,
        num_residual_blocks=num_residual_blocks,
    )

    output = model(arange_4d_image)

    assert output.image.shape == arange_4d_image.shape


def test_hific_autoencoder_compress_decompress(arange_4d_image):
    model = HiFiCAutoencoder(in_channels=arange_4d_image.shape[1])

    model.update()

    compressed = model.compress(arange_4d_image)
    decompressed = model.decompress(compressed)

    assert decompressed.shape == arange_4d_image.shape


def test_hific_autoencoder_collect_parameters():
    model = HiFiCAutoencoder()

    model_parameters, quantile_parameters = model.collect_parameters()

    manual_model_parameters = [
        p
        for n, p in model.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    ]
    manual_quantile_parameters = [
        p
        for n, p in model.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    ]

    for p1, p2 in zip(model_parameters, manual_model_parameters):
        assert torch.equal(p1, p2)

    for p1, p2 in zip(quantile_parameters, manual_quantile_parameters):
        assert torch.equal(p1, p2)


@pytest.mark.parametrize("mode", ["forward", "compress"])
def test_update_tensor_devices(mode):
    model = HiFiCAutoencoder()

    model.update_tensor_devices(mode)


def test_freeze_encoder():
    model = HiFiCAutoencoder()

    model.freeze_encoder()

    for p in model.encoder.parameters():
        assert p.requires_grad is False

    model.train()

    for p in model.encoder.parameters():
        assert p.requires_grad is False


def test_freeze_bottleneck():
    model = HiFiCAutoencoder()

    model.freeze_bottleneck()

    for module in [
        model.hyper_bottleneck,
        model.hyper_synthesis_mean,
        model.hyper_synthesis_scale,
        model.hyper_analysis,
        model.latent_bottleneck,
    ]:
        for p in module.parameters():
            assert p.requires_grad is False

    model.train()

    for module in [
        model.hyper_bottleneck,
        model.hyper_synthesis_mean,
        model.hyper_synthesis_scale,
        model.hyper_analysis,
        model.latent_bottleneck,
    ]:
        for p in module.parameters():
            assert p.requires_grad is False
