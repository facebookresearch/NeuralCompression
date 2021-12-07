# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List, Sequence, Tuple

import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule
from torch import Tensor

from neuralcompression.models import ScaleHyperprior


class ScaleHyperpriorLightning(LightningModule):
    """
    Model and training loop for the scale hyperprior model.

    Combines a pre-defined scale hyperprior model with its training loop
    for use with PyTorch Lightning.

    Args:
        model: the ScaleHyperprior model to train.
        distortion_lambda: A scaling factor for the distortion term
            of the loss.
        learning_rate: passed to the main network optimizer (i.e. the one that
            adjusts the analysis and synthesis parameters).
        aux_learning_rate: passed to the optimizer that learns the quantiles
            used to build the CDF table for the entropy codder.
    """

    def __init__(
        self,
        model: ScaleHyperprior,
        distortion_lambda: float = 1e-2,
        learning_rate: float = 1e-3,
        aux_learning_rate: float = 1e-3,
    ):
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate
        self.aux_learning_rate = aux_learning_rate
        self.distortion_lambda = distortion_lambda

    def forward(self, images):
        return self.model(images)

    def rate_distortion_loss(
        self,
        reconstruction: Tensor,
        latent_likelihoods: Tensor,
        hyper_latent_likelihoods: Tensor,
        original: Tensor,
    ):
        num_images, _, height, width = original.shape
        num_pixels = num_images * height * width

        bits = (
            latent_likelihoods.log().sum() + hyper_latent_likelihoods.log().sum()
        ) / -math.log(2)
        bpp_loss = bits / num_pixels

        distortion_loss = F.mse_loss(reconstruction, original)
        combined_loss = self.distortion_lambda * 255 ** 2 * distortion_loss + bpp_loss

        return bpp_loss, distortion_loss, combined_loss

    def update(self, force=True):
        return self.model.update(force=force)

    def compress(
        self, images: Tensor
    ) -> Tuple[List[str], List[str], Sequence[int], Sequence[int], Sequence[int]]:
        return self.model.compress(images)

    def decompress(
        self,
        y_strings: List[str],
        z_strings: List[str],
        image_shape: Sequence[int],
        y_shape: Sequence[int],
        z_shape: Sequence[int],
    ):
        return self.model.decompress(
            y_strings, z_strings, image_shape, y_shape, z_shape
        )

    def training_step(self, batch, batch_idx, optimizer_idx):

        if optimizer_idx not in [0, 1]:
            raise ValueError(
                f"Received unexpected optimizer index {optimizer_idx}"
                " - should be 0 or 1"
            )

        if optimizer_idx == 0:
            x_hat, y_likelihoods, z_likelihoods = self(batch)
            bpp_loss, distortion_loss, combined_loss = self.rate_distortion_loss(
                x_hat, y_likelihoods, z_likelihoods, batch
            )
            self.log_dict(
                {
                    "bpp_loss": bpp_loss.item(),
                    "distortion_loss": distortion_loss.item(),
                    "loss": combined_loss.item(),
                },
                sync_dist=True,
            )
            return combined_loss

        else:
            # This is the loss for learning the quantiles of the
            # distribution for the hyperprior.
            quantile_loss = self.model.quantile_loss()
            self.log("quantile_loss", quantile_loss.item(), sync_dist=True)
            return quantile_loss

    def validation_step(self, batch, batch_idx):
        x_hat, y_likelihoods, z_likelihoods = self(batch)
        bpp_loss, distortion_loss, combined_loss = self.rate_distortion_loss(
            x_hat, y_likelihoods, z_likelihoods, batch
        )

        self.log_dict(
            {
                "val_loss": combined_loss.item(),
                "val_distortion_loss": distortion_loss.item(),
                "val_bpp_loss": bpp_loss.item(),
            },
            sync_dist=True,
        )

    def configure_optimizers(self):
        model_param_dict, quantile_param_dict = self.model.collect_parameters()

        optimizer = optim.Adam(
            model_param_dict.values(),
            lr=self.learning_rate,
        )
        aux_optimizer = optim.Adam(
            quantile_param_dict.values(),
            lr=self.aux_learning_rate,
        )

        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

        return (
            {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_loss"},
            },
            {"optimizer": aux_optimizer},
        )
