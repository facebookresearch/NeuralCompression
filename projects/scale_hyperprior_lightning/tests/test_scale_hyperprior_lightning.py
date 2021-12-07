# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from neuralcompression.models import ScaleHyperprior
from projects.scale_hyperprior_lightning.scale_hyperprior import (
    ScaleHyperpriorLightning,
)


@pytest.mark.parametrize(
    "network_channels,compression_channels,img_size,batch_size",
    [(32, 64, 128, 1), (32, 64, 128, 4)],
)
def test_hyperprior_training(
    network_channels, compression_channels, img_size, batch_size
):
    # Tests the training and validation loop of the PTL lightning module.
    #
    # Tests that the scale hyperprior's PyTorch LightningModule (which
    # is responsible for the training and validation loop logic, logging,
    # etc.) can complete several training and validation steps without
    # crashing.

    train_ds = [torch.randn(3, img_size, img_size) for _ in range(10)]
    val_ds = [torch.randn(3, img_size, img_size) for _ in range(10)]

    train_dl = DataLoader(train_ds, batch_size=batch_size)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    module = ScaleHyperprior(
        network_channels=network_channels, compression_channels=compression_channels
    )

    lightning_module = ScaleHyperpriorLightning(module)

    trainer = Trainer(fast_dev_run=3)
    trainer.fit(lightning_module, train_dataloader=train_dl, val_dataloaders=val_dl)
