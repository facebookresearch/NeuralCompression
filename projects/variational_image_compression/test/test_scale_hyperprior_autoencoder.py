# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import neuralcompression.models

from ..lightning import ScaleHyperpriorAutoencoder


class TestScaleHyperpriorAutoencoder:
    def test__init__(self):
        network_channels = 128
        compression_channels = 192
        in_channels = 3
        distortion_trade_off = 1e-2

        lightning_module = ScaleHyperpriorAutoencoder(
            network_channels,
            compression_channels,
            in_channels,
            distortion_trade_off,
        )

        assert isinstance(
            lightning_module.network,
            neuralcompression.models.ScaleHyperpriorAutoencoder,
        )

        assert lightning_module.network.network_channels == network_channels
        assert lightning_module.network.compression_channels == compression_channels
        assert lightning_module.network.in_channels == in_channels

        assert lightning_module.hparams.distortion_trade_off == distortion_trade_off

    def test_configure_optimizers(self):
        network_channels = 128
        compression_channels = 192
        in_channels = 3
        distortion_trade_off = 1e-2

        lightning_module = ScaleHyperpriorAutoencoder(
            network_channels,
            compression_channels,
            in_channels,
            distortion_trade_off,
        )

        configured_optimizers = lightning_module.configure_optimizers()

        optimizer, bottleneck_optimizer = configured_optimizers

        assert isinstance(
            optimizer["optimizer"],
            Adam,
        )

        optimizer_lr_scheduler = optimizer["lr_scheduler"]

        assert optimizer_lr_scheduler["monitor"] == "val_rate_distortion"

        assert isinstance(
            optimizer_lr_scheduler["scheduler"],
            ReduceLROnPlateau,
        )

        assert isinstance(
            bottleneck_optimizer["optimizer"],
            Adam,
        )

        with pytest.raises(KeyError):
            _ = bottleneck_optimizer["lr_scheduler"]

    def test_forward(self):
        network_channels = 128
        compression_channels = 192
        in_channels = 3
        distortion_trade_off = 1e-2

        lightning_module = ScaleHyperpriorAutoencoder(
            network_channels,
            compression_channels,
            in_channels,
            distortion_trade_off,
        )

        x = torch.rand(16, 3, 64, 64)

        x_hat, probabilities = lightning_module.forward(x)

        assert x_hat.shape == (16, 3, 64, 64)

        y_probabilities, z_probabilities = probabilities

        assert y_probabilities.shape == (16, 192, 4, 4)
        assert z_probabilities.shape == (16, 128, 1, 1)

    def test_training_step(self):
        network_channels = 128
        compression_channels = 192
        in_channels = 3
        distortion_trade_off = 1e-2

        lightning_module = ScaleHyperpriorAutoencoder(
            network_channels,
            compression_channels,
            in_channels,
            distortion_trade_off,
        )

        batch = torch.rand(16, 3, 64, 64)

        loss = lightning_module.training_step(batch, 0, 0)

        assert isinstance(loss, Tensor)

        bottleneck_loss = lightning_module.training_step(batch, 0, 1)

        assert isinstance(bottleneck_loss, Tensor)

    def test_validation_step(self):
        network_channels = 128
        compression_channels = 192
        in_channels = 3
        distortion_trade_off = 1e-2

        lightning_module = ScaleHyperpriorAutoencoder(
            network_channels,
            compression_channels,
            in_channels,
            distortion_trade_off,
        )

        batch = torch.rand(16, 3, 64, 64)

        loss = lightning_module.validation_step(batch, 0)

        assert isinstance(loss, Tensor)
