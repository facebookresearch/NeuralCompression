# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import NamedTuple, Optional, Tuple

import torch
import torch.optim as optim
import torchmetrics
from _utils import (
    DvcStage,
    DvcStage1,
    DvcStage2,
    DvcStage3,
    DvcStage4and5,
    LossFunctions,
)
from compressai.zoo import models
from pytorch_lightning import LightningModule
from torch import Tensor

import neuralcompression.functional as ncF
from neuralcompression.models import DVC

TRAINING_STAGES = {
    "1_motion_estimation": DvcStage1,
    "2_motion_compression": DvcStage2,
    "3_motion_compensation": DvcStage3,
    "4_total_2frame": DvcStage4and5,
    "5_total": DvcStage4and5,
}


class LoggingMetrics(NamedTuple):
    gop_total_loss: Tensor
    gop_distortion_loss: Tensor
    gop_bpp: Tensor
    gop_flow_bpp: Optional[Tensor]
    gop_residual_bpp: Optional[Tensor]


class DvcModule(LightningModule):
    """
    Model and training loop for the DVC model.

    Combines a pre-defined DVC model with its training loop for use with
    PyTorch Lightning.

    Args:
        model: The DVC model to train.
        training_stage: Current stage of training process. One of
            ``("1_motion_estimation", "2_motion_compression",
            "3_motion_compensation", "4_total")``. See DVC paper for details.
        pretrained_model_name: Name of model from CompressAI model zoo to use
            for compressing I-frames.
        pretrained_model_quality_level: Quality level of model from CompressAI.
        num_pframes: Number of P-frames to process for training.
        distortion_type: Type of distortion loss function. Must be from
            ``("MSE")``.
        distortion_lambda: A scaling factor for the distortion term of the
            loss.
        learning_rate: passed to the main network optimizer (i.e. the one that
            adjusts the analysis and synthesis parameters).
        aux_learning_rate: passed to the optimizer that learns the quantiles
            used to build the CDF table for the entropy codder.
        lr_scheduler_params: Used for ``StepLR``, specify ``step_size`` and
            ``gamma``.
    """

    def __init__(
        self,
        model: DVC,
        training_stage: str,
        pretrained_model_name: str,
        pretrained_model_quality_level: int,
        num_pframes: int = 1,
        distortion_type: str = "MSE",
        distortion_lambda: float = 256.0,
        learning_rate: float = 1e-4,
        aux_learning_rate: float = 1e-3,
        lr_scheduler_params: Optional[Tuple[int, float]] = None,
        grad_clip_value: float = 1.0,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.training_stage = training_stage
        self.learning_rate = learning_rate
        self.num_pframes = num_pframes
        self.aux_learning_rate = aux_learning_rate
        self.distortion_lambda = distortion_lambda
        self.lr_scheduler_params = lr_scheduler_params
        self.grad_clip_value = grad_clip_value
        self.iframe_model = models[pretrained_model_name](
            pretrained_model_quality_level, pretrained=True
        )

        # make sure training stage is valid
        if training_stage not in TRAINING_STAGES.keys():
            raise ValueError(f"training stage {training_stage} not recognized.")
        if distortion_type == "MSE":
            distortion_loss = torch.nn.MSELoss()
        else:
            raise ValueError(f"distortion_type {distortion_type} not recognized.")

        # set up loss functions
        def distortion_fn(image1, image2):
            return self.distortion_lambda * distortion_loss(image1, image2)

        def entropy_fn(probabilities, reduction_factor):
            return (
                ncF.information_content(probabilities, reduction="sum")
                / reduction_factor
            )

        self.loss_functions = LossFunctions(distortion_fn, entropy_fn)

        # set up model - this includes functions for computing losses and
        # compressing
        self.model: DvcStage = TRAINING_STAGES[training_stage](
            model, num_pframes, self.loss_functions
        )

        # metrics
        self.psnr = torchmetrics.PSNR(data_range=1.0)

    def recompose_model(self, model: DVC):
        """Used by trainer to rebuild the model after a training run."""
        return self.model.recompose_model(model)

    def forward(self, images):
        return self.model(images)

    def update(self, force=True):
        return self.model.update(force=force)

    def compress_iframe(self, batch):
        """Compress an I-frame using a pretrained model."""
        self.iframe_model = self.iframe_model.eval()
        self.iframe_model.update()
        with torch.no_grad():
            output = self.iframe_model(batch[:, 0])

        batch[:, 0] = output["x_hat"]
        num_pixels = batch.shape[0] * batch.shape[-2] * batch.shape[-1]
        bpp_loss = sum(
            self.loss_functions.entropy_fn(likelihoods, num_pixels)
            for likelihoods in output["likelihoods"].values()
        )

        return batch, bpp_loss

    def compute_loss_and_metrics(self, loss_values, current_metrics):
        # loss function collection
        combined_bpp = 0
        loss = loss_values.distortion_loss
        gop_distortion_loss = (
            current_metrics.gop_distortion_loss + loss_values.distortion_loss.detach()
        )

        if loss_values.flow_entropy_loss is not None:
            loss = loss + loss_values.flow_entropy_loss
            combined_bpp = combined_bpp + loss_values.flow_entropy_loss.detach()
            gop_flow_bpp: Optional[Tensor] = (
                current_metrics.gop_flow_bpp + loss_values.flow_entropy_loss.detach()
            )
        else:
            gop_flow_bpp = None

        if loss_values.resid_entropy_loss is not None:
            loss = loss + loss_values.resid_entropy_loss
            combined_bpp = combined_bpp + loss_values.resid_entropy_loss.detach()
            gop_residual_bpp: Optional[Tensor] = (
                current_metrics.gop_residual_bpp
                + loss_values.resid_entropy_loss.detach()
            )
        else:
            gop_residual_bpp = None

        gop_total_loss = current_metrics.gop_total_loss + loss.detach()
        gop_bpp = current_metrics.gop_bpp + combined_bpp

        return loss, LoggingMetrics(
            gop_total_loss=gop_total_loss,
            gop_distortion_loss=gop_distortion_loss,
            gop_bpp=gop_bpp,
            gop_flow_bpp=gop_flow_bpp,
            gop_residual_bpp=gop_residual_bpp,
        )

    def log_all_metrics(
        self, log_key, reduction, image2_list, image2_est_list, logging_metrics
    ):
        self.log(f"{log_key}gop_loss", logging_metrics.gop_total_loss / reduction)
        self.log(
            f"{log_key}gop_distortion_loss",
            logging_metrics.gop_distortion_loss / reduction,
        )
        self.log(f"{log_key}gop_bpp", logging_metrics.gop_bpp / reduction)
        if logging_metrics.gop_flow_bpp is not None:
            self.log(f"{log_key}gop_flow_bpp", logging_metrics.gop_flow_bpp / reduction)
        if logging_metrics.gop_residual_bpp is not None:
            self.log(
                f"{log_key}gop_residual_bpp",
                logging_metrics.gop_residual_bpp / reduction,
            )

        # logging metrics
        self.psnr(torch.cat(image2_list), torch.cat(image2_est_list))
        self.log(f"{log_key}psnr", self.psnr)

    def training_step(self, batch, batch_idx):
        log_key = f"{self.training_stage}/train_"

        if isinstance(self.optimizers(), list):
            [opt1, opt2] = self.optimizers()
        else:
            opt1 = self.optimizers()
            opt2 = None

        # compress the iframe and get its bpp cost (no grads)
        batch, iframe_bpp = self.compress_iframe(batch)

        # update main model params
        # gop = "Group of Pictures"
        logging_metrics = LoggingMetrics(
            gop_total_loss=0,
            gop_distortion_loss=0,
            gop_bpp=iframe_bpp,
            gop_flow_bpp=0,
            gop_residual_bpp=0,
        )
        image2_list = []
        image2_est_list = []

        image1 = batch[:, 0]
        for i in range(self.num_pframes):
            opt1.zero_grad()  # we backprop for every P-frame
            image2 = batch[:, i + 1]
            loss_values, images = self.model.compute_batch_loss(image1, image2)
            image1 = images.image2_est  # images are detached

            # keep track of these for other distortion metrics
            # note: these have no grads
            image2_list.append(images.image2)
            image2_est_list.append(images.image2_est)

            # loss function collection
            loss, logging_metrics = self.compute_loss_and_metrics(
                loss_values, logging_metrics
            )

            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(
                opt1.param_groups[0]["params"], self.grad_clip_value
            )
            opt1.step()

        # lr step
        if self.lr_schedulers() is not None:
            self.lr_schedulers().step()

        # stat reductions and logging
        reduction = self.num_pframes + 1
        self.log_all_metrics(
            log_key, reduction, image2_list, image2_est_list, logging_metrics
        )

        # auxiliary update
        # this is the loss for learning the quantiles of the bottlenecks.
        if opt2 is not None:
            opt2.zero_grad()
            aux_loss = self.model.quantile_loss()
            self.log(f"{log_key}quantile_loss", aux_loss, sync_dist=True)
            self.manual_backward(aux_loss)
            opt2.step()

    def validation_step(self, batch, batch_idx):
        log_key = f"{self.training_stage}/val_"
        # gop = "Group of Pictures"
        logging_metrics = LoggingMetrics(
            gop_total_loss=0,
            gop_distortion_loss=0,
            gop_bpp=0,
            gop_flow_bpp=0,
            gop_residual_bpp=0,
        )
        image2_list = []
        image2_est_list = []

        batch, gop_bpp = self.compress_iframe(batch)  # bpp_total w/o grads
        image1 = batch[:, 0]
        for i in range(self.num_pframes):
            image2 = batch[:, i + 1]
            loss_values, images = self.model.compute_batch_loss(image1, image2)
            image1 = images.image2_est  # images are detached

            # keep track of these for other distortion metrics
            image2_list.append(images.image2)
            image2_est_list.append(images.image2_est)

            # loss function collection
            loss, logging_metrics = self.compute_loss_and_metrics(
                loss_values, logging_metrics
            )

        # stat reductions and logging
        reduction = self.num_pframes + 1
        self.log("val_loss", loss)
        self.log_all_metrics(
            log_key, reduction, image2_list, image2_est_list, logging_metrics
        )

    def configure_optimizers(self):
        # we have to train the model and the entropy bottleneck quantiles
        # separately
        model_param_dict, quantile_param_dict = self.model.collect_parameters()

        base_optim = optim.Adam(model_param_dict.values(), lr=self.learning_rate)
        if self.lr_scheduler_params is not None:
            scheduler = optim.lr_scheduler.StepLR(
                base_optim, **self.lr_scheduler_params
            )
            optimizers = [
                {
                    "optimizer": base_optim,
                    "lr_scheduler": scheduler,
                }
            ]
        else:
            optimizers = [{"optimizer": base_optim}]

        if quantile_param_dict is not None:
            optimizers.append(
                {
                    "optimizer": optim.Adam(
                        quantile_param_dict.values(),
                        lr=self.aux_learning_rate,
                    )
                }
            )

        return optimizers
