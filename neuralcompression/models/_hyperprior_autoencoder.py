# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional, OrderedDict, Union

import torch
import torch.nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from torch import Tensor
from torch.nn import Module

from neuralcompression.layers import (
    HyperAnalysisTransformation2D,
    HyperSynthesisTransformation2D,
)
from ._prior_autoencoder import PriorAutoencoder


class HyperpriorAutoencoder(PriorAutoencoder):
    """Base class for implementing prior autoencoder architectures.

    The class composes a bottleneck module (e.g. the ``EntropyBottleneck``
    module provided by the CompressAI package) with an autoencoder (i.e.
    encoder and decoder modules).

    Using the base class is as straightforward as inheriting from the class and
    defining an ``encoder_module`` and ``decoder_module``. You may optionally
    provide a ``hyper_encoder_module`` and ``hyper_decoder_module`` (e.g. for
    implementing hyperprior architectures). The ``neuralcompression.layers``
    package includes a standard encoder (``AnalysisTransformation2D``), decoder
    (``SynthesisTransformation2D``), hyper encoder
    (``HyperAnalysisTransformation2D``), and hyper decoder
    (``HyperSynthesisTransformation2D``).

    Args:
        network_channels: number of channels in the network.
        compression_channels: number of inferred latent compression features.
        in_channels:
        hyper_encoder:
        hyper_decoder:
        minimum:
        maximum:
        steps:
    """

    hyper_encoder: Module
    hyper_decoder: Module

    def __init__(
        self,
        network_channels: int = 128,
        compression_channels: int = 192,
        in_channels: int = 3,
        hyper_encoder: Optional[Module] = None,
        hyper_decoder: Optional[Module] = None,
        minimum: Union[int, float] = 0.11,
        maximum: Union[int, float] = 256,
        steps: int = 64,
    ):
        super(HyperpriorAutoencoder, self).__init__(
            network_channels,
            compression_channels,
            in_channels,
            bottleneck=EntropyBottleneck(network_channels),
        )

        if hyper_encoder is not None:
            self.hyper_encoder = hyper_encoder
        else:
            self.hyper_encoder = HyperAnalysisTransformation2D(
                network_channels,
                compression_channels,
                in_channels,
            )

        if hyper_decoder is not None:
            self.hyper_decoder = hyper_decoder
        else:
            self.hyper_decoder = HyperSynthesisTransformation2D(
                network_channels,
                compression_channels,
                in_channels,
            )

        self.minimum = math.log(minimum)
        self.maximum = math.log(maximum)

        self.steps = steps

        self.gaussian_conditional = GaussianConditional(None)

    @classmethod
    def from_state_dict(
        cls,
        state_dict: OrderedDict,
    ):
        """
        Args:
            state_dict:

        Returns:
        """
        network_channels = state_dict["encoder.encode.0.weight"].size()[0]

        compression_channels = state_dict["encoder.encode.6.weight"].size()[0]

        hyperprior = cls(network_channels, compression_channels)

        hyperprior.load_state_dict(state_dict)

        return hyperprior

    def load_state_dict(
        self,
        state_dict: OrderedDict[str, Tensor],
        strict: bool = True,
    ):
        """
        Args:
            state_dict:
            strict:
        """
        bottleneck_buffer_names = [
            "_quantized_cdf",
            "_offset",
            "_cdf_length",
            "scale_table",
        ]

        for bottleneck_buffer_name in bottleneck_buffer_names:
            name = f"gaussian_conditional.{bottleneck_buffer_name}"

            size = state_dict[name].size()

            registered_buffers = []

            for name, buffer in self.gaussian_conditional.named_buffers():
                if name == bottleneck_buffer_name:
                    registered_buffers += [buffer]

            if registered_buffers:
                registered_buffer = registered_buffers[0]

                if registered_buffer.numel() == 0:
                    registered_buffer.resize_(size)

        super(HyperpriorAutoencoder, self).load_state_dict(state_dict)

    def scales(self) -> Tensor:
        """
        Returns:
        """
        return torch.exp(torch.linspace(self.minimum, self.maximum, self.steps))

    def update_bottleneck(
        self,
        force: bool = False,
        scales: Optional[Tensor] = None,
    ) -> bool:
        """
        Args:
            force:
            scales:

        Returns:
        """
        if scales is None:
            scales = self.scales()

        updated = self.gaussian_conditional.update_scale_table(scales, force=force)

        updated |= super(HyperpriorAutoencoder, self).update_bottleneck(force=force)

        return updated
