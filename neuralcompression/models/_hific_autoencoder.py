# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from torch import Tensor

import neuralcompression.functional as ncF
from neuralcompression import HyperpriorCompressedOutput, HyperpriorOutput

from ._hific_encoder_decoder import HiFiCEncoder, HiFiCGenerator
from ._hyperprior_autoencoder import HyperpriorAutoencoderBase


def _conv(
    cin: int,
    cout: int,
    kernel_size: int = 5,
    stride: int = 2,
) -> nn.Conv2d:
    return nn.Conv2d(
        cin, cout, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2
    )


def _deconv(
    cin: int,
    cout: int,
    kernel_size: int = 5,
    stride: int = 2,
) -> nn.ConvTranspose2d:
    return nn.ConvTranspose2d(
        cin,
        cout,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=1,
        padding=kernel_size // 2,
    )


class HiFiCAutoencoder(HyperpriorAutoencoderBase):
    """
    HiFiC autoencoder architecture.

    This implements the HiFiC autoencoder as described in the following paper:

    High-Fidelity Generative Image Compression
    F. Mentzer, G. Toderici, M. Tschannen, E. Agustsson

    Args:
        in_channels: The number of channels in the input image.
        latent_features: The number of features to use as input to the
            conditional Gaussian layer.
        hyper_feautures: The number of features to use for the hyperprior.
        num_residual_blocks: Number of residual blocks to use at the start of
            generator decoding.
        freeze_encoder: Whether to freeze training of the encoder.
        freeze_bottleneck: Whether to freeze training of the bottleneck (both
            conditional Gaussian and hyperprior).
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_features: int = 220,
        hyper_features: int = 320,
        num_residual_blocks: int = 9,
        freeze_encoder: bool = False,
        freeze_bottleneck: bool = False,
    ):
        super().__init__()
        self._factor = 64  # this is the full downsampling factor
        self._frozen_encoder = False
        self._frozen_bottleneck = False
        self.logger = logging.getLogger(self.__class__.__name__)
        self.encoder = HiFiCEncoder(
            in_channels=in_channels, latent_features=latent_features
        )
        self.decoder = HiFiCGenerator(
            image_channels=in_channels,
            latent_features=latent_features,
            n_residual_blocks=num_residual_blocks,
        )
        self.hyper_analysis = nn.Sequential(
            _conv(latent_features, hyper_features, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            _conv(hyper_features, hyper_features),
            nn.ReLU(inplace=True),
            _conv(hyper_features, hyper_features),
        )
        self.hyper_synthesis_mean = nn.Sequential(
            _deconv(hyper_features, hyper_features),
            nn.ReLU(inplace=True),
            _deconv(hyper_features, hyper_features),
            nn.ReLU(inplace=True),
            _conv(
                hyper_features,
                latent_features,
                stride=1,
                kernel_size=3,
            ),
        )
        self.hyper_synthesis_scale = nn.Sequential(
            _deconv(hyper_features, hyper_features),
            nn.ReLU(inplace=True),
            _deconv(hyper_features, hyper_features),
            nn.ReLU(inplace=True),
            _conv(
                hyper_features,
                latent_features,
                stride=1,
                kernel_size=3,
            ),
        )
        self.hyper_bottleneck = EntropyBottleneck(hyper_features)
        self.latent_bottleneck = GaussianConditional(scale_table=None)

        if freeze_encoder:
            self.freeze_encoder()
        if freeze_bottleneck:
            self.freeze_bottleneck()

        self.set_compress_cpu_layers(
            [
                self.hyper_bottleneck,
                self.latent_bottleneck,
                self.hyper_synthesis_scale,
                self.hyper_synthesis_mean,
                self.hyper_analysis,
            ]
        )

    @property
    def factor(self) -> int:
        return self._factor

    @property
    def frozen_encoder(self) -> bool:
        return self._frozen_encoder

    @property
    def frozen_bottleneck(self) -> bool:
        return self._frozen_bottleneck

    def update_tensor_devices(self, target_operation: str):
        """
        Updates location of model weights based on target_operation.

        Args:
            target_operation: Either ''forward'' or ''compress''. For
                ''forward'', all tensors will be located on the model device.
                For ''compress'', weights that are used for likelihood
                calculation will be held on the CPU.
        """
        if target_operation not in ("forward", "compress"):
            raise ValueError("Target operation must be 'forward' or 'compress'.")

        if target_operation == "forward":
            target_device = self.encoder.blocks[0][0].weight.device
            self.to(target_device)
        else:
            self.update()
            self._set_devices_for_compress()

        self._device_setting = target_operation

    def _set_devices_for_compress(self):
        cpu = torch.device("cpu")
        self.hyper_bottleneck.to(cpu)
        self.latent_bottleneck.to(cpu)
        self.hyper_synthesis_scale.to(cpu)
        self.hyper_synthesis_mean.to(cpu)
        self.hyper_analysis.to(cpu)

    def _check_compress_devices(self) -> bool:
        result = True
        cpu = torch.device("cpu")
        for module in [
            self.hyper_bottleneck,
            self.latent_bottleneck,
            self.hyper_analysis,
            self.hyper_synthesis_mean,
            self.hyper_synthesis_scale,
        ]:
            for param in module.parameters():
                if not param.device == cpu:
                    result = False

        return result

    def freeze_encoder(self):
        """Freeze and disable training for the encoder."""
        self._frozen_encoder = True
        self.logger.info("Freezing encoder!")
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad_(False)

    def freeze_bottleneck(self):
        """Freeze and disable training for the bottleneck."""
        module: nn.Module
        self._frozen_bottleneck = True
        self.logger.info("Freezing bottleneck!")
        for module in [
            self.hyper_bottleneck,
            self.hyper_synthesis_mean,
            self.hyper_synthesis_scale,
            self.hyper_analysis,
            self.latent_bottleneck,
        ]:
            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)

    def train(self, mode: bool = True) -> "HiFiCAutoencoder":
        model = super().train(mode)
        if model._frozen_encoder:
            model.freeze_encoder()
        if model._frozen_bottleneck:
            model.freeze_bottleneck()

        return model

    def forward(self, image: Tensor) -> HyperpriorOutput:
        if not self.device_setting == "forward":
            raise RuntimeError(
                "Must call update_tensor_devices('forward') to use model forward."
            )

        # encode image to latent
        latent = self.encoder(image)

        # bottleneck processing
        hyper_latent = self.hyper_analysis(latent)
        hyper_latent, hyper_likelihoods = self.hyper_bottleneck(hyper_latent)
        with torch.no_grad():
            _, quantized_hyper_latent_likelihoods = self.hyper_bottleneck(
                hyper_latent, training=False
            )
        means = self.hyper_synthesis_mean(hyper_latent)
        scales = self.hyper_synthesis_scale(hyper_latent)
        quantized_latents, latent_likelihoods = self.latent_bottleneck(
            latent, scales, means=means
        )
        with torch.no_grad():
            _, quantized_latent_likelihoods = self.latent_bottleneck(
                latent, scales, means=means, training=False
            )
        if self.training:
            # we use straight-through to train the generator
            latent = self._ste_quantize(latent, means)
        else:
            # means we're in eval mode and the latents have been rounded
            latent = quantized_latents

        # reconstruct the image
        reconstruction = self.decoder(latent)

        return HyperpriorOutput(
            image=reconstruction,
            latent=latent,
            latent_likelihoods=latent_likelihoods,
            quantized_latent_likelihoods=quantized_latent_likelihoods.detach(),
            hyper_latent=hyper_latent,
            hyper_latent_likelihoods=hyper_likelihoods,
            quantized_hyper_latent_likelihoods=quantized_hyper_latent_likelihoods.detach(),
        )

    def compress(
        self, image: Tensor, force_cpu: bool = True
    ) -> HyperpriorCompressedOutput:
        """
        Compress a batch of images into strings.

        Args:
            image: Tensor to compress (in [0, 1] floating point range).
            force_cpu: Whether to throw an error if any operations are not on
                CPU.
        """
        if not self._on_cpu():
            if force_cpu:
                raise ValueError("All params must be on CPU if force_cpu=True.")

            if not self._check_compress_devices():
                raise ValueError(
                    "Some layers on GPU that should be on CPU. Call "
                    "update_tensor_devices('compress') to use partial-GPU "
                    "compression."
                )

        image, (height, width) = ncF.pad_image_to_factor(image, self._factor)

        # image analysis
        latent = self.encoder(image).cpu()

        # hyper analysis
        hyper_latent = self.hyper_analysis(latent)

        # hyper bottleneck
        hyper_latent_strings = self.hyper_bottleneck.compress(hyper_latent)
        hyper_latent_decoded = self.hyper_bottleneck.decompress(
            hyper_latent_strings, hyper_latent.shape[-2:]
        )

        # hyper synthesis
        means = self.hyper_synthesis_mean(hyper_latent_decoded)
        scales = self.hyper_synthesis_scale(hyper_latent_decoded)

        # latent compression
        indexes = self.latent_bottleneck.build_indexes(scales)
        latent_strings = self.latent_bottleneck.compress(latent, indexes, means=means)

        return HyperpriorCompressedOutput(
            latent_strings=latent_strings,
            hyper_latent_strings=hyper_latent_strings,
            image_size=(height, width),
            padded_size=(image.shape[-2], image.shape[-1]),
        )

    def decompress(
        self, compressed_data: HyperpriorCompressedOutput, force_cpu: bool = True
    ) -> Tensor:
        """
        Decompress a batch of images from strings.

        Args:
            compressed_data: Strings of data to decompress.
            force_cpu: Whether to throw an error if any operations are not on
                CPU.
        """
        if not self._on_cpu():
            if force_cpu:
                raise ValueError("All params must be on CPU if force_cpu=True.")

            if not self._check_compress_devices():
                raise ValueError(
                    "Some layers on GPU that should be on CPU. Call "
                    "update_tensor_devices('compress') to use partial-GPU "
                    "compression."
                )

            device = self.decoder.blocks[-1].weight.device
        else:
            device = torch.device("cpu")

        latent_size = (
            compressed_data.padded_size[0] // 2**4,
            compressed_data.padded_size[1] // 2**4,
        )
        hyper_latent_size = (latent_size[0] // 2**2, latent_size[1] // 2**2)

        # hyper synthesis
        hyper_latent_decoded = self.hyper_bottleneck.decompress(
            compressed_data.hyper_latent_strings, hyper_latent_size
        )
        means = self.hyper_synthesis_mean(hyper_latent_decoded)
        scales = self.hyper_synthesis_scale(hyper_latent_decoded)

        # image synthesis
        indexes = self.latent_bottleneck.build_indexes(scales)
        latent_decoded = self.latent_bottleneck.decompress(
            compressed_data.latent_strings, indexes, means=means
        )
        reconstruction: Tensor = self.decoder(latent_decoded.to(device))

        return reconstruction[
            :, :, : compressed_data.image_size[0], : compressed_data.image_size[1]
        ].clamp(0.0, 1.0)
