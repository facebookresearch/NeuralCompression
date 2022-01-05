# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, List, Tuple

import torch
import torch.nn.functional as F
from compressai.entropy_models import EntropyBottleneck
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch import Size, Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import neuralcompression.models


class _PriorAutoencoder(LightningModule):
    network: neuralcompression.models.PriorAutoencoder

    def __init__(
        self,
        distortion_trade_off: float = 1e-2,
        optimizer_lr: float = 1e-3,
        bottleneck_optimizer_lr: float = 1e-3,
    ):
        super(_PriorAutoencoder, self).__init__()

        self.save_hyperparameters()

    def bottleneck_loss(self) -> Tensor:
        if isinstance(self.network.bottleneck, EntropyBottleneck):
            return self.network.bottleneck.loss()
        else:
            return torch.tensor(0.0)

    def compress(self, bottleneck: Tensor) -> Tuple[List[List[str]], Size]:
        return self.network.compress(bottleneck)  # type: ignore

    def configure_optimizers(self):
        parameters, bottleneck_parameters = self.network.group_parameters()

        optimizer = Adam(
            parameters.values(),
            self.hparams.optimizer_lr,
        )

        bottleneck_optimizer = Adam(
            bottleneck_parameters.values(),
            self.hparams.bottleneck_optimizer_lr,
        )

        return (
            {
                "lr_scheduler": {
                    "scheduler": ReduceLROnPlateau(optimizer, "min"),
                    "monitor": "val_rate_distortion",
                },
                "optimizer": optimizer,
            },
            {
                "optimizer": bottleneck_optimizer,
            },
        )

    def decompress(self, strings: List[str], broadcast_size: Size) -> Tensor:
        return self.network.decompress(strings, broadcast_size)  # type: ignore

    def forward(self, *args, **kwargs) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        x: Tensor

        (x,) = args

        return self.network(x)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return super(_PriorAutoencoder, self).predict_dataloader()

    def rate_distortion_loss(
        self,
        x_hat: Tensor,
        x: Tensor,
        likelihoods: Tuple[Tensor, ...],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        n, _, h, w = x.shape

        pixels = -math.log(2) * (n * h * w)

        bpps = []

        for likelihood in likelihoods:
            bpps += [float(likelihood.log().sum()) / pixels]

        rate = Tensor(bpps).sum()

        distortion = F.mse_loss(x_hat, x)

        rate_distortion = (
            self.hparams.distortion_trade_off * 255 ** 2 * (rate + distortion)  # type: ignore
        )

        return rate, distortion, rate_distortion

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return super(_PriorAutoencoder, self).test_dataloader()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return super(_PriorAutoencoder, self).train_dataloader()

    def training_step(self, *args, **kwargs) -> Tensor:
        batch: Tensor
        batch_idx: int
        optimizer_idx: int

        batch, batch_idx, optimizer_idx = args

        if optimizer_idx == 0:
            x_hat, likelihoods = self(batch)

            rate, distortion, rate_distortion = self.rate_distortion_loss(
                x_hat,
                batch,
                likelihoods,
            )

            dictionary = {
                "rate": rate.item(),
                "distortion": distortion.item(),
                "rate_distortion": rate_distortion.item(),
            }

            self.log_dict(dictionary, sync_dist=True)

            return rate_distortion
        else:
            bottleneck_loss = self.bottleneck_loss()

            self.log(
                "bottleneck_loss",
                bottleneck_loss.item(),
                sync_dist=True,
            )

            return bottleneck_loss

    def update_bottleneck(self, force: bool = True):
        return self.network.update_bottleneck(force)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return super(_PriorAutoencoder, self).val_dataloader()

    def validation_step(self, *args, **kwargs) -> Tensor:
        batch: Any
        batch_idx: int

        batch, batch_idx = args

        x_hat, likelihoods = self(batch)

        rate, distortion, rate_distortion = self.rate_distortion_loss(
            x_hat,
            batch,
            likelihoods,
        )

        dictionary = {
            "val_rate": rate.item(),
            "val_distortion": distortion.item(),
            "val_rate_distortion": rate_distortion.item(),
        }

        self.log_dict(dictionary, sync_dist=True)

        return rate_distortion
