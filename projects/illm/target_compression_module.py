# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List, NamedTuple, Optional

import hydra
import torch
from image_module import ImageModule
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import Dataset

from neuralcompression import HyperpriorOutput
from neuralcompression.loss_fn import DistortionLoss, MSELoss, TargetRateConfig
from neuralcompression.metrics import pickle_size_of
from neuralcompression.models import HyperpriorAutoencoderBase


class RateLossOutput(NamedTuple):
    rate_loss: Tensor
    total_bpp: Tensor
    quantized_total_bpp: Tensor
    latent_bpp: Tensor
    quantized_latent_bpp: Tensor
    hyper_bpp: Tensor
    quantized_hyper_bpp: Tensor
    target: float
    lam2: float


class TargetRateCompressionModule(ImageModule):
    def __init__(
        self,
        model: HyperpriorAutoencoderBase,
        target_rate_config: TargetRateConfig,
        optimizer_config: DictConfig,
        distortion_lam: float = 1.0,
        distortion_loss: Optional[DistortionLoss] = None,
        gradient_clip_val: Optional[float] = None,
        gradient_clip_algorithm: str = "norm",
    ):
        super().__init__()
        self.model: HyperpriorAutoencoderBase = model
        self.target_rate_config = target_rate_config
        self.optimizer_config = optimizer_config
        self.distortion_lam = distortion_lam
        self.train_dataset: Optional[Dataset] = None
        self.eval_dataset: Optional[Dataset] = None

        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm

        if distortion_loss is None:
            distortion_loss = MSELoss()
        self.distortion_loss = distortion_loss

        self.automatic_optimization = False

    def forward(self, x):
        return self.model(x)

    def on_load_checkpoint(self, checkpoint) -> None:
        output = super().on_load_checkpoint(checkpoint)
        self.reset_all_metrics()
        return output

    def on_train_epoch_start(self) -> None:
        # just make sure we clear any buffers from training metrics
        self.reset_all_metrics()

    def _get_scheduled_rate_param(
        self, base_param: float, factors: List[float], steps: List[int]
    ) -> float:
        for factor, sched_step in zip(factors, steps):
            param = base_param * factor
            if self.trainer.global_step < sched_step:
                return param

        return base_param * factors[-1]

    def _calc_bits_per_batch(self, likelihoods: Tensor) -> Tensor:
        batch_size = likelihoods.shape[0]
        likelihoods = likelihoods.view(batch_size, -1)
        return likelihoods.log().sum(1) / -math.log(2)

    def rate_loss(
        self,
        original: Tensor,
        latent_likelihoods: Tensor,
        quantized_latent_likelihoods: Tensor,
        hyper_latent_likelihoods: Tensor,
        quantized_hyper_latent_likelihoods: Tensor,
    ) -> RateLossOutput:
        # calculate bits-per-pixel for both quantized and noisy quantization
        _, _, height, width = original.shape
        num_pixels = height * width

        latent_bpp = self._calc_bits_per_batch(latent_likelihoods) / num_pixels
        quantized_latent_bpp = (
            self._calc_bits_per_batch(quantized_latent_likelihoods) / num_pixels
        )
        hyper_bpp = self._calc_bits_per_batch(hyper_latent_likelihoods) / num_pixels
        quantized_hyper_bpp = (
            self._calc_bits_per_batch(quantized_hyper_latent_likelihoods) / num_pixels
        )
        total_bpp = latent_bpp + hyper_bpp
        quantized_total_bpp = quantized_latent_bpp + quantized_hyper_bpp

        # apply rate targeting for loss
        target = self._get_scheduled_rate_param(
            self.target_rate_config.target_bpp,
            self.target_rate_config.target_factors,
            self.target_rate_config.target_steps,
        )
        lam2 = self._get_scheduled_rate_param(
            self.target_rate_config.lam_levels[1],
            self.target_rate_config.lam2_factors,
            self.target_rate_config.lam2_steps,
        )
        lams = torch.where(
            quantized_total_bpp > target, lam2, self.target_rate_config.lam_levels[0]
        )

        return RateLossOutput(
            rate_loss=(lams * total_bpp).mean(),
            total_bpp=total_bpp.detach().mean(),
            quantized_total_bpp=quantized_total_bpp.detach().mean(),
            latent_bpp=latent_bpp.detach().mean(),
            quantized_latent_bpp=quantized_latent_bpp.detach().mean(),
            hyper_bpp=hyper_bpp.detach().mean(),
            quantized_hyper_bpp=quantized_hyper_bpp.detach().mean(),
            target=target,
            lam2=lam2,
        )

    def training_step(self, batch, batch_idx):
        images: Tensor
        output: HyperpriorOutput
        model_opt, quantile_opt = self.optimizers()
        model_sched, quantile_sched = self.lr_schedulers()

        ########### MODEL OPTIMIZER ###########
        images = batch

        output = self.model(images)

        distortion_loss = self.distortion_loss(output.image, images)
        rate_output = self.rate_loss(
            original=images,
            latent_likelihoods=output.latent_likelihoods,
            quantized_latent_likelihoods=output.quantized_latent_likelihoods,
            hyper_latent_likelihoods=output.hyper_latent_likelihoods,
            quantized_hyper_latent_likelihoods=output.quantized_hyper_latent_likelihoods,
        )
        loss_train = self.distortion_lam * distortion_loss + rate_output.rate_loss

        model_opt.zero_grad()
        self.manual_backward(loss_train)
        if self.gradient_clip_val is not None:
            self.clip_gradients(
                model_opt,
                gradient_clip_val=self.gradient_clip_val,
                gradient_clip_algorithm=self.gradient_clip_algorithm,
            )
        model_opt.step()
        model_sched.step()

        ########### QUANTILE OPTIMIZER ###########
        quantile_loss = self.model.hyper_bottleneck.loss()

        quantile_opt.zero_grad()
        self.manual_backward(quantile_loss)
        quantile_opt.step()
        quantile_sched.step()

        ########### LOGGING ###########
        self.log_quality_metrics(output, images, prefix="train")
        self.log_dict(
            {
                "train/total_loss": loss_train,
                "train/distortion_loss": distortion_loss,
                "train/rate_loss": rate_output.rate_loss,
                "train/noisy_total_bpp": rate_output.total_bpp,
                "train/quantized_total_bpp": rate_output.quantized_total_bpp,
                "train/noisy_hyper_bpp": rate_output.hyper_bpp,
                "train/quantized_hyper_bpp": rate_output.quantized_hyper_bpp,
                "train/noisy_latent_bpp": rate_output.latent_bpp,
                "train/quantized_latent_bpp": rate_output.quantized_latent_bpp,
                "train/target": rate_output.target,
                "train/lam2": rate_output.lam2,
                "train/quantile_loss": quantile_loss,
            }
        )

    def eval_step(self, batch, batch_idx, prefix: str):
        images: Tensor
        output: HyperpriorOutput

        images = batch

        output = self.model(images)

        distortion_loss = self.distortion_loss(output.image, images)
        rate_output = self.rate_loss(
            original=images,
            latent_likelihoods=output.latent_likelihoods,
            quantized_latent_likelihoods=output.quantized_latent_likelihoods,
            hyper_latent_likelihoods=output.hyper_latent_likelihoods,
            quantized_hyper_latent_likelihoods=output.quantized_hyper_latent_likelihoods,
        )
        eval_loss = self.distortion_lam * distortion_loss + rate_output.rate_loss

        self.log_quality_metrics(output, images, prefix=prefix)
        self.log_dict(
            {
                f"{prefix}/total_loss": eval_loss,
                f"{prefix}/distortion_loss": distortion_loss,
                f"{prefix}/rate_loss": rate_output.rate_loss,
                f"{prefix}/total_bpp": rate_output.total_bpp,
                f"{prefix}/quantized_total_bpp": rate_output.quantized_total_bpp,
                f"{prefix}/hyper_bpp": rate_output.hyper_bpp,
                f"{prefix}/quantized_hyper_bpp": rate_output.quantized_hyper_bpp,
                f"{prefix}/latent_bpp": rate_output.latent_bpp,
                f"{prefix}/quantized_latent_bpp": rate_output.quantized_latent_bpp,
                f"{prefix}/target": rate_output.target,
                f"{prefix}/lam2": rate_output.lam2,
            },
            sync_dist=True,
        )

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")

    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        self.model.update()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        prefix = "test"
        images: Tensor
        if self.test_dataset_name is None:
            raise ValueError("Must run module.update_test_dataset_name()")

        self.model.update_tensor_devices("compress")

        dataset_name = self.test_dataset_name

        if isinstance(batch, Tensor):
            images = batch
        else:
            images, _ = batch

        compressed = self.model.compress(images, force_cpu=False)
        decompressed = self.model.decompress(compressed, force_cpu=False)

        test_loss = self.distortion_loss(decompressed, images)

        num_bytes = pickle_size_of(compressed)
        bpp = num_bytes * 8 / (images.shape[0] * images.shape[-2] * images.shape[-1])

        self.log_dict(
            {
                f"{prefix}/{dataset_name}_bpp": bpp,
                f"{prefix}/{dataset_name}_test_distortion": test_loss,
            }
        )
        self.log_quality_metrics(
            decompressed, images, prefix=prefix, dataset_name=dataset_name
        )

        return test_loss

    def configure_optimizers(self):
        model_params, quant_params = self.model.collect_parameters()

        optimizer = hydra.utils.instantiate(
            self.optimizer_config.model_opt, params=model_params
        )
        aux_optimizer = hydra.utils.instantiate(
            self.optimizer_config.aux_opt, params=quant_params
        )

        lr_sched = hydra.utils.instantiate(
            self.optimizer_config.model_opt_schedule, optimizer=optimizer
        )
        aux_lr_sched = hydra.utils.instantiate(
            self.optimizer_config.aux_opt_schedule, optimizer=aux_optimizer
        )

        return ([optimizer, aux_optimizer], [lr_sched, aux_lr_sched])
