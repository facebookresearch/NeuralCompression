# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
from target_compression_module import TargetRateCompressionModule
from torch import Tensor

from neuralcompression import HyperpriorOutput
from neuralcompression.loss_fn import (
    DiscriminatorLoss,
    DistortionLoss,
    GeneratorLoss,
    TargetRateConfig,
)
from neuralcompression.models import HyperpriorAutoencoderBase


class GANCompressionModule(TargetRateCompressionModule):
    def __init__(
        self,
        model: HyperpriorAutoencoderBase,
        target_rate_config: TargetRateConfig,
        discriminator: nn.Module,
        discriminator_loss: DiscriminatorLoss,
        generator_loss: GeneratorLoss,
        optimizer_config: DictConfig,
        distortion_lam: float = 1.0,
        distortion_loss: Optional[DistortionLoss] = None,
        latent_projector: Optional[nn.Module] = None,
        generator_weight: float = 1.0,
        gradient_clip_val: Optional[float] = None,
        gradient_clip_algorithm: str = "norm",
        mc_sampling: bool = True,
    ):
        super().__init__(
            model=model,
            target_rate_config=target_rate_config,
            optimizer_config=optimizer_config,
            distortion_lam=distortion_lam,
            distortion_loss=distortion_loss,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )

        self.discriminator = discriminator
        self.discriminator_loss = discriminator_loss
        self.latent_projector = latent_projector
        self.generator_loss = generator_loss
        self.generator_weight = generator_weight
        self.mc_sampling = mc_sampling

    def disable_gradients(self, opt):
        for group in opt.param_groups:
            for param in group["params"]:
                param.requires_grad_(False)

    def enable_gradients(self, opt):
        for group in opt.param_groups:
            for param in group["params"]:
                param.requires_grad_(True)

    def run_discriminator(self, images: Tensor, context: Tensor) -> Tensor:
        if self.discriminator.is_conditional:
            return self.discriminator(images, context)
        else:
            return self.discriminator(images)

    def forward(self, x):
        return self.model(x)

    def _resample_batch(
        self, full_images: Tensor, mc_sampling: bool = False
    ) -> Tuple[Tensor, Tensor]:
        if mc_sampling is True:
            split_ind = full_images.shape[0] // 2
            model_images = full_images[:split_ind]
            disc_images = full_images[split_ind : split_ind * 2]
        else:
            model_images = full_images
            disc_images = full_images

        return model_images, disc_images

    def training_step(self, batch, batch_idx):
        output: HyperpriorOutput
        opts = self.optimizers()
        scheds = self.lr_schedulers()
        if len(opts) == 3:
            model_opt, disc_opt, quantile_opt = opts
            model_sched, disc_sched, quantile_sched = scheds
        else:
            model_opt, disc_opt = opts
            model_sched, disc_sched = scheds
            quantile_opt = None

        # if doing Monte Carlo sampling, split the batch for the discriminator
        model_images, disc_images = self._resample_batch(
            batch, mc_sampling=self.mc_sampling
        )

        # get the discriminator/generator targets
        if self.latent_projector is not None:
            with torch.no_grad():
                fake_target = self.latent_projector(model_images)
                real_target = self.latent_projector(disc_images)
        else:
            real_target = torch.tensor(1, dtype=torch.long)
            fake_target = torch.tensor(1, dtype=torch.long)

        ########### GENERATOR STEP ###########
        self.disable_gradients(disc_opt)
        self.enable_gradients(model_opt)

        output = self.model(model_images)

        # distortion loss
        distortion_loss = self.distortion_loss(output.image, model_images)

        # generator loss
        g_loss = self.generator_loss(
            self.run_discriminator(output.image, output.latent.detach()), fake_target
        )

        # rate loss
        rate_output = self.rate_loss(
            original=model_images,
            latent_likelihoods=output.latent_likelihoods,
            quantized_latent_likelihoods=output.quantized_latent_likelihoods,
            hyper_latent_likelihoods=output.hyper_latent_likelihoods,
            quantized_hyper_latent_likelihoods=output.quantized_hyper_latent_likelihoods,
        )

        # total loss
        model_loss = (
            self.distortion_lam * distortion_loss
            + rate_output.rate_loss
            + self.generator_weight * g_loss
        )

        # step optimizer
        model_opt.zero_grad()
        self.manual_backward(model_loss)
        if self.gradient_clip_val is not None:
            self.clip_gradients(
                model_opt,
                gradient_clip_val=self.gradient_clip_val,
                gradient_clip_algorithm=self.gradient_clip_algorithm,
            )
        model_opt.step()
        model_sched.step()

        ########### DISCRIMINATOR STEP ###########
        self.disable_gradients(model_opt)
        self.enable_gradients(disc_opt)

        if self.mc_sampling:
            with torch.no_grad():
                disc_output = self.model(disc_images)
        else:
            disc_output = output

        fake_logits = self.discriminator(output.image.detach(), output.latent.detach())
        real_logits = self.discriminator(disc_images, disc_output.latent.detach())

        disc_loss = self.discriminator_loss(
            real_logits=real_logits, fake_logits=fake_logits, target=real_target
        )

        # step optimizer
        disc_opt.zero_grad()
        self.manual_backward(disc_loss)
        if self.gradient_clip_val is not None:
            self.clip_gradients(
                disc_opt,
                gradient_clip_val=self.gradient_clip_val,
                gradient_clip_algorithm=self.gradient_clip_algorithm,
            )
        disc_opt.step()
        disc_sched.step()

        ########### QUANTILE OPTIMIZER ###########
        if quantile_opt is not None:
            quantile_loss = self.model.hyper_bottleneck.loss()

            quantile_opt.zero_grad()
            self.manual_backward(quantile_loss)
            quantile_opt.step()
            quantile_sched.step()
        else:
            quantile_loss = 0.0

        ########### LOGGING ###########
        self.log_quality_metrics(output, model_images, prefix="train")
        self.log_dict(
            {
                "train/total_model_loss": model_loss,
                "train/distortion_loss": distortion_loss,
                "train/rate_loss": rate_output.rate_loss,
                "train/generator_loss": g_loss,
                "train/discriminator_loss": disc_loss,
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
        if self.latent_projector is not None:
            target = self.latent_projector(images)
        else:
            target = torch.tensor(1)

        # distortion loss
        distortion_loss = self.distortion_loss(output.image, images)

        # generator loss
        g_loss = self.generator_loss(
            self.discriminator(output.image, output.latent), target
        )

        # rate loss
        rate_output = self.rate_loss(
            original=images,
            latent_likelihoods=output.latent_likelihoods,
            quantized_latent_likelihoods=output.quantized_latent_likelihoods,
            hyper_latent_likelihoods=output.hyper_latent_likelihoods,
            quantized_hyper_latent_likelihoods=output.quantized_hyper_latent_likelihoods,
        )

        total_loss = (
            self.distortion_lam * distortion_loss
            + rate_output.rate_loss
            + self.generator_weight * g_loss
        )

        # logging
        self.log_quality_metrics(output, images, prefix=prefix)
        self.log_dict(
            {
                f"{prefix}/total_model_loss": total_loss,
                f"{prefix}/distortion_loss": distortion_loss,
                f"{prefix}/rate_loss": rate_output.rate_loss,
                f"{prefix}/generator_loss": g_loss,
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

    def configure_optimizers(self):
        model_params, quant_params = self.model.collect_parameters()

        optimizer = hydra.utils.instantiate(
            self.optimizer_config.model_opt, params=model_params
        )
        discriminator_optimizer = hydra.utils.instantiate(
            self.optimizer_config.discriminator_opt,
            params=self.discriminator.parameters(),
        )
        if len(quant_params) > 0:
            aux_optimizer = hydra.utils.instantiate(
                self.optimizer_config.aux_opt, params=quant_params
            )
        else:
            aux_optimizer = None

        lr_sched = hydra.utils.instantiate(
            self.optimizer_config.model_opt_schedule, optimizer=optimizer
        )
        discriminator_sched = hydra.utils.instantiate(
            self.optimizer_config.discriminator_opt_schedule,
            optimizer=discriminator_optimizer,
        )
        if aux_optimizer is not None:
            aux_lr_sched = hydra.utils.instantiate(
                self.optimizer_config.aux_opt_schedule, optimizer=aux_optimizer
            )

            return (
                [optimizer, discriminator_optimizer, aux_optimizer],
                [lr_sched, discriminator_sched, aux_lr_sched],
            )
        else:
            return (
                [optimizer, discriminator_optimizer],
                [lr_sched, discriminator_sched],
            )
