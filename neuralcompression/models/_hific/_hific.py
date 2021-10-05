"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import functools
import typing

import lpips
import torch
import torch.nn
import torch.nn.functional

from ._discriminator_data import _DiscriminatorData
from ._distortion_loss import _distortion_loss
from ._generative_adversarial_loss import _generative_adversarial_loss
from ._hific_discriminator import HiFiCDiscriminator
from ._hific_encoder import HiFiCEncoder
from ._hific_generator import HiFiCGenerator
from ._hific_kind import _HiFiCKind
from ._hific_mode import _HiFiCMode
from ._intermediate_data import _IntermediateData
from ._pad_image import _pad_image
from ._weighted_rate_loss import _weighted_rate_loss
from .. import ScaleHyperprior


class HiFiC(torch.nn.Module):
    _discriminate: bool = False

    def __init__(
        self,
        encoder: typing.Optional[torch.nn.Module] = None,
        prior: typing.Optional[torch.nn.Module] = None,
        generator: typing.Optional[torch.nn.Module] = None,
        discriminator: typing.Optional[torch.nn.Module] = None,
        mode: str = "train",
        kind: str = "compression",
        standardize_image: bool = False,
        beta: float = 0.15,
        k_m: float = 0.002,
        k_p: float = 1.0,
        ignore_schedule: bool = False,
    ):
        super(HiFiC, self).__init__()

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = HiFiCEncoder()

        if prior is not None:
            self.prior = prior
        else:
            self.prior = ScaleHyperprior()

        if generator is not None:
            self.generator = generator
        else:
            self.generator = HiFiCGenerator()

        if self._discriminate:
            if discriminator is not None:
                self.discriminator = discriminator
            else:
                self.discriminator = HiFiCDiscriminator()

            self.generative_adversarial_loss = functools.partial(
                _generative_adversarial_loss,
                "non_saturating",
            )
        else:
            self.discriminator = None

        self.mode = _HiFiCMode(mode)

        self.kind = _HiFiCKind(kind)

        self.standardize_image = standardize_image

        self.beta = beta

        self.k_m = k_m

        self.k_p = k_p

        self.ignore_schedule = ignore_schedule

        self.step_counter = 0

        if mode == _HiFiCMode.EVALUATE:
            self.evaluate = True
        else:
            self.evaluate = False

            if self.kind == _HiFiCKind.GENERATIVE_ADVERSARIAL_COMPRESSION:
                self._discriminate = True

        self.perceptual_loss = lpips.LPIPS()

        if torch.cuda.is_available():
            self.perceptual_loss.cuda()

    def forward(
        self,
        x: torch.Tensor,
        train_generator: bool = False,
    ):
        losses = {}

        if train_generator is True:
            self.step_counter += 1

        image_dimensions = tuple(x.size()[1:])

        if self.mode == _HiFiCMode.EVALUATE and not self.training:
            x = _pad_image(
                x,
                x.size()[2:],
                2 ** self.encoder.n_downsampling_layers,
            )

        y = self.encoder(x)

        if self.mode == _HiFiCMode.EVALUATE and not self.training:
            y = _pad_image(
                y,
                y.size()[2:],
                2 ** self.prior.analysis_net.n_downsampling_layers,
            )

        _, quantized_latent_features, _ = self.prior(y)

        synthetic_image = self.generator(quantized_latent_features)

        if self.standardize_image:
            synthetic_image = torch.tanh(synthetic_image)

        if self.mode == _HiFiCMode.EVALUATE and not self.training:
            synthetic_image = synthetic_image[
                ..., : image_dimensions[1], : image_dimensions[2]
            ]

        intermediate_data = _IntermediateData(
            x,
            synthetic_image,
            quantized_latent_features,
            0.0,  # FIXME: get nbpp from prior
            0.0,  # FIXME: get qbpp from prior
        )

        authentic_image = intermediate_data.authentic_image
        synthetic_image = intermediate_data.synthetic_image

        if self.mode == _HiFiCMode.EVALUATE:
            if self.standardize_image:
                synthetic_image = (synthetic_image + 1.0) / 2.0

            synthetic_image = torch.clamp(synthetic_image, 0.0, 1.0)

            return synthetic_image, intermediate_data.qbpp

        if self.standardize_image:
            authentic_image = (authentic_image + 1.0) / 2.0
            synthetic_image = (synthetic_image + 1.0) / 2.0

        distortion_loss = _distortion_loss(
            authentic_image,
            synthetic_image,
        )

        perceptual_loss = torch.mean(
            self.perceptual_loss.forward(
                authentic_image,
                synthetic_image,
                normalize=True,
            )
        )

        weighted_distortion_loss = self.k_m * distortion_loss
        weighted_perceptual_loss = self.k_p * perceptual_loss

        weighted_rate_loss, rate_penalty = _weighted_rate_loss(
            intermediate_data.nbpp,
            intermediate_data.qbpp,
            self.step_counter,
            ignore_schedule=self.ignore_schedule,
        )

        compression_loss = weighted_rate_loss + weighted_distortion_loss

        compression_loss += weighted_perceptual_loss

        if self._discriminate:
            if not train_generator:
                synthetic_image = synthetic_image.detach()

            discriminated, discriminated_predictions = self.discriminator(
                torch.cat([authentic_image, synthetic_image]),
                torch.repeat_interleave(
                    intermediate_data.quantized_latent_features.detach(),
                    repeats=2,
                    dim=0,
                ),
            )

            discriminator_data = _DiscriminatorData(
                *torch.chunk(
                    torch.squeeze(discriminated),
                    chunks=2,
                ),
                *torch.chunk(
                    torch.squeeze(discriminated_predictions),
                    chunks=2,
                )
            )

            discriminator_loss = self.generative_adversarial_loss(
                discriminator_data,
                mode="discriminator",
            )

            generator_loss = self.generative_adversarial_loss(
                discriminator_data,
                mode="generator",
            )

            compression_loss += self.beta * generator_loss

            losses["discriminator_loss"] = discriminator_loss

        losses["compression_loss"] = compression_loss

        return losses, intermediate_data
