# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tvtF
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN as GeneralizedDivisiveNormalization
from compressai.models.google import get_scale_table
from compressai.models.utils import update_registered_buffers
from torch import Tensor
from torch.nn.parameter import Parameter


def _conv(
    cin: int,
    cout: int,
    kernel_size: int,
    stride: int = 1,
) -> nn.Conv2d:
    return nn.Conv2d(
        cin, cout, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2
    )


def _deconv(
    cin: int,
    cout: int,
    kernel_size: int,
    stride: int = 1,
) -> nn.ConvTranspose2d:
    return nn.ConvTranspose2d(
        cin,
        cout,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )


def _resize(x: Tensor, target_shape: Sequence[int]) -> Tensor:
    """
    Resizes a tensor to the target shape.

    Given a tensor of shape [batch_size, num_channels, height, width] and
    a [height, width] target_shape, this function resizes the input
    tensor to the taget shape using center cropping if the input is larger
    than the target shape, and using nearest neighbour interpolation if
    the input is smaller than the target shape.

    Args:
        x: the tensor to resize, of shape [batch_size, num_channels,
            height, width].
        target_shape: a [height, width] pair to reshape the input to.

    Returns:
        the resized tensor.
    """

    height, width = x.shape[2:]
    target_height, target_width = target_shape

    if height >= target_height and width >= target_width:
        return tvtF.center_crop(x, target_shape)
    elif height <= target_height and width <= target_width:
        return F.interpolate(x, target_shape, mode="nearest")
    else:
        raise ValueError(
            f"Input tensor (with shape {x.shape}) is larger than the"
            f" target shape of {target_shape} along one height/width axis"
            " and is smaller than the target shape along the other axis."
        )


class _AbsoluteValue(nn.Module):
    def forward(self, inp: Tensor) -> Tensor:
        return torch.abs(inp)


class ScaleHyperpriorImageAnalysis(nn.Module):
    """
    Image analysis network for the scale hyperprior model.

    Args:
        network_channels, compression_channels: see ScaleHyperprior
    """

    def __init__(self, network_channels: int, compression_channels: int):
        super().__init__()
        self.model = nn.Sequential(
            _conv(3, network_channels, kernel_size=5, stride=2),
            GeneralizedDivisiveNormalization(network_channels),
            _conv(network_channels, network_channels, kernel_size=5, stride=2),
            GeneralizedDivisiveNormalization(network_channels),
            _conv(network_channels, network_channels, kernel_size=5, stride=2),
            GeneralizedDivisiveNormalization(network_channels),
            _conv(network_channels, compression_channels, kernel_size=5, stride=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class ScaleHyperpriorImageSynthesis(nn.Module):
    """
    Image synthesis network for the scale hyperprior model.

    Args:
        network_channels, compression_channels: see ScaleHyperprior
    """

    def __init__(self, network_channels: int, compression_channels: int):
        super().__init__()
        self.model = nn.Sequential(
            _deconv(compression_channels, network_channels, kernel_size=5, stride=2),
            GeneralizedDivisiveNormalization(network_channels, inverse=True),
            _deconv(network_channels, network_channels, kernel_size=5, stride=2),
            GeneralizedDivisiveNormalization(network_channels, inverse=True),
            _deconv(network_channels, network_channels, kernel_size=5, stride=2),
            GeneralizedDivisiveNormalization(network_channels, inverse=True),
            _deconv(network_channels, 3, kernel_size=5, stride=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class ScaleHyperpriorHyperAnalysis(nn.Module):
    """
    Hyper analysis network for the scale hyperprior model.

    Args:
        network_channels, compression_channels: see ScaleHyperprior
    """

    def __init__(self, network_channels: int, compression_channels: int):
        super().__init__()
        self.model = nn.Sequential(
            _AbsoluteValue(),
            _conv(compression_channels, network_channels, kernel_size=3),
            nn.ReLU(inplace=True),
            _conv(network_channels, network_channels, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            _conv(network_channels, network_channels, kernel_size=5, stride=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class ScaleHyperpriorHyperSynthesis(nn.Module):
    """
    Hyper synthesis network for the scale hyperprior model.

    Args:
        network_channels, compression_channels: see ScaleHyperprior
    """

    def __init__(self, network_channels: int, compression_channels: int):
        super().__init__()
        self.model = nn.Sequential(
            _deconv(network_channels, network_channels, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            _deconv(network_channels, network_channels, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            _deconv(network_channels, compression_channels, kernel_size=3),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class ScaleHyperprior(nn.Module):
    """
    Implementation of the scale hyperprior compression model.

    An implementation of the architecture introduced in
    "Variational image compression with a scale hyperprior"
    by Johannes Ballé, David Minnen, Saurabh Singh,
    Sung Jin Hwang, Nick Johnston. https://arxiv.org/abs/1802.01436.

    This model has six subcomponents (two analysis networks, two synthesis
    networks, and two bottlenecks), which can be configured in two different
    ways. For maximum customization, each subcomponent can be passed as an
    argument in the model constructor (detailed below).

    Alternatively, some or none of the subcomponents can be specified
    and default components following the architecture of the paper will be
    used. In order for these defaults to be created, the network_channels
    and compression_channels arguments may need to be passed, in order to
    determine the sizes of the subcomponents to create. Specifically, if
    an analysis or synthesis network is unspecified, both channels
    arguments need to be passed, whereas if the hyper_bottleneck module
    is unspecified, then only network_channels needs to be passed (no
    arguments need to be passed to construct the default image_bottleck
    module).

    TODO: Add proper types for bottleneck component args instead of using
    nn.Module, to enforce that the appropriate compress/decompress
    functions exist.

    Args:
        network_channels: the number of convolutional channels to use in
            each layer of the analysis and synthesis networks if these
            components are unspecified below. Denoted with the letter
            N in the hyperprior paper.
        compression_channels: the number of convolutional channels in the
            latent to be quantized (defines the number of channels in the
            last layer of the image analysis network and last layer of the
            hyper synthesis network if these components are unspecified
            below). Denoted with the letter M in the hyperprior paper.
        image_analysis: an nn.Module to transform the input image into a
            latent to compress.
        image_synthesis: an nn.Module to transform the noisy/quantized image
            latent into a reconstruction of the original image.
        image_bottleneck: an nn.Module to be the entropy bottleneck layer
            for the image latent.
        hyper_analysis: an nn.Module to transform the input latent into a
            hyper latent to compress.
        hyper_synthesis: an nn.Module to transform the noisy/quantized hyper
            latent into the scales tensor used in the image_bottleneck.
        hyper_bottleneck: an nn.Module to be the entropy bottleneck layer for
            the hyper latent.
    """

    def __init__(
        self,
        network_channels: Optional[int] = None,
        compression_channels: Optional[int] = None,
        image_analysis: Optional[nn.Module] = None,
        image_synthesis: Optional[nn.Module] = None,
        image_bottleneck: Optional[nn.Module] = None,
        hyper_analysis: Optional[nn.Module] = None,
        hyper_synthesis: Optional[nn.Module] = None,
        hyper_bottleneck: Optional[nn.Module] = None,
    ):
        super().__init__()

        if (
            None
            in [
                image_analysis,
                image_synthesis,
                hyper_analysis,
                hyper_analysis,
                hyper_synthesis,
            ]
            and None in [network_channels, compression_channels]
        ):
            raise ValueError(
                "When one or more analysis or synthesis networks is unspecified, "
                "'network_channels' and 'compressions_channels' must be"
                " passed."
            )

        if hyper_bottleneck is None and network_channels is None:
            raise ValueError(
                "When hyper_bottleneck is unspecified, "
                "'network_channels' must be passed."
            )

        if image_analysis is not None:
            self.image_analysis = image_analysis
        else:
            assert network_channels is not None and compression_channels is not None
            self.image_analysis = ScaleHyperpriorImageAnalysis(
                network_channels, compression_channels
            )

        if hyper_analysis is not None:
            self.hyper_analysis = hyper_analysis
        else:
            assert network_channels is not None and compression_channels is not None
            self.hyper_analysis = ScaleHyperpriorHyperAnalysis(
                network_channels, compression_channels
            )

        if hyper_synthesis is not None:
            self.hyper_synthesis = hyper_synthesis
        else:
            assert network_channels is not None and compression_channels is not None
            self.hyper_synthesis = ScaleHyperpriorHyperSynthesis(
                network_channels, compression_channels
            )

        if image_synthesis is not None:
            self.image_synthesis = image_synthesis
        else:
            assert network_channels is not None and compression_channels is not None
            self.image_synthesis = ScaleHyperpriorImageSynthesis(
                network_channels, compression_channels
            )

        if image_bottleneck is not None:
            self.image_bottleneck = image_bottleneck
        else:
            self.image_bottleneck = GaussianConditional(scale_table=None)

        if hyper_bottleneck is not None:
            self.hyper_bottleneck = hyper_bottleneck
        else:
            self.hyper_bottleneck = EntropyBottleneck(channels=network_channels)

    def forward(self, images: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Calculates an image's reconstruction and compression likelihoods.

        This function is for use during model training. Given a batch of
        input images, this function sends the images through the model's
        encoder and decoder, using additive uniform noise as a stand-in
        for quantization so the model is fully differentiable.

        In addition to returning the decoder's reconstructed image,
        this function also outputs the likelihoods assigned by the
        entropy bottleneck layers to the corresponding quantized latents.
        This is needed when computing the model's bits-per-pixel loss.

        Args:
            images: a batch of images with shape [batch size, channels,
                height, width]

        Returns:
            A tuple of:
                1.  The decoder's image reconstruction (this has the same
                    shape as the original input).
                2.  The likelihoods assigned by the image bottleneck layer
                    to the quantized image latent.
                3.  The likelihoods assigned by the hyper bottleneck layer
                    to the quantized hyper latent.
        """

        latent = self.image_analysis(images)
        hyper_latent = self.hyper_analysis(latent)
        noisy_hyper_latent, hyper_latent_likelihoods = self.hyper_bottleneck(
            hyper_latent
        )
        scales = self.hyper_synthesis(noisy_hyper_latent)

        if scales.shape != latent.shape:
            scales = _resize(scales, latent.shape[2:])

        noisy_latent, latent_likelihoods = self.image_bottleneck(latent, scales)
        reconstruction = self.image_synthesis(noisy_latent)

        if reconstruction.shape != images.shape:
            reconstruction = _resize(reconstruction, images.shape[2:])

        return reconstruction, latent_likelihoods, hyper_latent_likelihoods

    def update(self, force=False):
        """
        Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            force: overwrite previous values (default: False)

        Returns:
            updated: True if one of the bottlenecks was updated.
        """

        if isinstance(self.image_bottleneck, GaussianConditional):
            image_bottleneck_updated = self.image_bottleneck.update_scale_table(
                get_scale_table(), force=force
            )
        else:
            image_bottleneck_updated = self.image_bottleneck.update(force=force)

        hyper_bottleneck_updated = self.hyper_bottleneck.update(force=force)
        return image_bottleneck_updated | hyper_bottleneck_updated

    def _on_cpu(self):
        cpu = torch.device("cpu")
        for param in self.parameters():
            if param.device != cpu:
                return False
        return True

    # TODO: Switch to named tuple
    def compress(
        self, images: Tensor, force_cpu: bool = True
    ) -> Tuple[List[str], List[str], Sequence[int], Sequence[int], Sequence[int]]:
        """
        Compress a batch of images into strings.

        Args:
            images: ``Tensor`` of shape [batch, channels, height, width].
            force_cpu: whether to throw an error if the model is
                not on the CPU when compressing. Compressing/decompressing
                on the GPU has known numerical and reproducability
                issues with the default entropy bottleneck implementation.

        Returns:
            latent_strings: list containing a compressed latent string for
                each image in the batch.
            hyper_latent_strings: list containing a compressed hyperprior
                string for each image in the batch.
            image_shape: list storing the height and width of the
                original images, for use during decoding.
            latent_shape: list storing the height and width of the
                image latent, for use during decoding.
            hyper_latent_shape: list storing the height and width of the
                hyperprior, for use during decoding.
        """
        if not self._on_cpu() and force_cpu:
            raise ValueError("Compress not supported on GPU.")

        latent = self.image_analysis(images)
        hyper_latent = self.hyper_analysis(latent)
        hyper_latent_strings = self.hyper_bottleneck.compress(hyper_latent)  # type: ignore
        hyper_latent_decoded = self.hyper_bottleneck.decompress(  # type: ignore
            hyper_latent_strings, hyper_latent.shape[2:]
        )
        scales = self.hyper_synthesis(hyper_latent_decoded)

        if scales.shape != latent.shape:
            scales = _resize(scales, latent.shape[2:])

        indexes = self.image_bottleneck.build_indexes(scales)  # type: ignore
        latent_strings = self.image_bottleneck.compress(latent, indexes)  # type: ignore
        return (
            latent_strings,
            hyper_latent_strings,
            images.shape[2:],
            latent.shape[2:],
            hyper_latent.shape[2:],
        )

    def decompress(
        self,
        latent_strings: List[str],
        hyper_latent_strings: List[str],
        image_shape: Sequence[int],
        latent_shape: Sequence[int],
        hyper_latent_shape: Sequence[int],
        force_cpu: bool = True,
    ) -> Tensor:
        """
        Decompress a batch of binary strings into images.

        Args:
            latent_strings: list containing a compressed latent string for
                each image in the batch.
            hyper_latent_strings: list containing a compressed hyperprior
                string for each image in the batch.
            image_shape: list storing the height and width of the
                original images.
            latent_shape: list storing the height and width of the
                image latent.
            hyper_latent_shape: list storing the height and width of
                the hyperprior.
            force_cpu: whether to throw an error if the model is
                not on the CPU when decompressing. Compressing/decompressing
                on the GPU has known numerical and reproducability
                issues with the default entropy bottleneck implementation.

        Returns:
            reconstruction: Tensor of shape [batch, channels, height, width].
        """
        if not self._on_cpu() and force_cpu:
            raise ValueError("Decompress not supported on GPU.")

        hyper_latent_decoded = self.hyper_bottleneck.decompress(  # type: ignore
            hyper_latent_strings, hyper_latent_shape
        )
        scales = self.hyper_synthesis(hyper_latent_decoded)

        if scales.shape[2:] != tuple(latent_shape):
            scales = _resize(scales, latent_shape)

        indexes = self.image_bottleneck.build_indexes(scales)  # type: ignore
        latent_decoded = self.image_bottleneck.decompress(latent_strings, indexes)  # type: ignore
        reconstruction = self.image_synthesis(latent_decoded).clamp_(0, 1)

        if reconstruction.shape[2:] != tuple(image_shape):
            reconstruction = _resize(reconstruction, image_shape)

        return reconstruction

    def load_state_dict(self, state_dict):
        """
        Updates the model's parameters from a saved dictionary.

        This model overrides the default load_state_dict implementation
        to properly load the CDF buffers of the entropy models.

        According to the original CompressAI implementation,
        these calls perform tensor resizing on load in order to
        handle variable-sized buffers.
        """
        if isinstance(self.hyper_bottleneck, EntropyBottleneck):
            update_registered_buffers(
                self.hyper_bottleneck,
                "hyper_bottleneck",
                ["_quantized_cdf", "_offset", "_cdf_length"],
                state_dict,
            )
        if isinstance(self.image_bottleneck, GaussianConditional):
            update_registered_buffers(
                self.image_bottleneck,
                "image_bottleneck",
                ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
                state_dict,
            )
        super().load_state_dict(state_dict)

    def collect_parameters(self) -> Tuple[Dict[str, Parameter], Dict[str, Parameter]]:
        """
        Separates the trainable parameters of the model into groups.

        The module's parameters are organized into "model parameters" (the
        parameters that dictate the function of the model) and "quantile
        parameters" (which are only used to learn the quantiles of the
        hyper_bottleneck layer's factorized distribution, for use at inference
        time).

        Returns:
            tuple of (model parameter_dict, quantile parameter_dict)
        """
        model_parameters = {
            n: p
            for n, p in self.named_parameters()
            if not n.endswith(".quantiles") and p.requires_grad
        }
        quantile_parameters = {
            n: p
            for n, p in self.named_parameters()
            if n.endswith(".quantiles") and p.requires_grad
        }

        model_keys = set(model_parameters.keys())
        quantile_keys = set(quantile_parameters.keys())

        # Make sure we don't have an intersection of parameters
        params_dict = dict(self.named_parameters())
        params_keys = set(params_dict.keys())

        inter_keys = model_keys.intersection(quantile_keys)
        union_keys = model_keys.union(quantile_keys)

        if len(inter_keys) != 0 or union_keys != params_keys:
            raise RuntimeError("Separating model and quantile parameters failed.")

        return model_parameters, quantile_parameters

    def quantile_loss(self) -> Tensor:
        """
        The loss to train the quantile parameters of bottleneck layers.
        """

        if isinstance(self.hyper_bottleneck, EntropyBottleneck):
            return self.hyper_bottleneck.loss()
        else:
            return torch.tensor(0.0)
