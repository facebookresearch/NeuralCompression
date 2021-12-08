# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, OrderedDict, Tuple

from compressai.entropy_models import EntropyBottleneck
from torch import Tensor
from torch.nn import Module, Parameter

from neuralcompression.layers import AnalysisTransformation2D, SynthesisTransformation2D


class PriorAutoencoder(Module):
    """Base class for implementing prior autoencoder architectures.
    The class composes a bottleneck module (e.g. the ``EntropyBottleneck``
    module provided by the CompressAI package) with an autoencoder (i.e.
    encoder and decoder modules).

    Using the base class is as straightforward as inheriting from the class and
    defining an ``encoder_module`` and ``decoder_module``. The
    ``neuralcompression.layers`` package includes a standard encoder
    (``AnalysisTransformation2D``) and decoder (``SynthesisTransformation2D``).

    Args:
        network_channels: number of channels in the network.
        compression_channels: number of inferred latent compression features.
        encoder: prior autoencoder encoder.
        decoder: prior autoencoder decoder.
        bottleneck: entropy bottleneck.
        bottleneck_name: name of entropy bottleneck.
        bottleneck_buffer_names: names of bottleneck buffers to persist.
    """

    hyper_encoder: Optional[Module]
    hyper_decoder: Optional[Module]

    def __init__(
        self,
        network_channels: int,
        compression_channels: int,
        in_channels: int = 3,
        encoder: Optional[Module] = None,
        decoder: Optional[Module] = None,
        bottleneck: Optional[EntropyBottleneck] = None,
        bottleneck_name: Optional[str] = None,
        bottleneck_buffer_names: Optional[List[str]] = None,
    ):
        super(PriorAutoencoder, self).__init__()

        self.network_channels = network_channels
        self.compression_channels = compression_channels
        self.in_channels = in_channels

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = AnalysisTransformation2D(
                self.network_channels,
                self.compression_channels,
                self.in_channels,
            )

        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder = SynthesisTransformation2D(
                self.network_channels,
                self.compression_channels,
                self.in_channels,
            )

        if bottleneck is not None:
            self.bottleneck = bottleneck
        else:
            self.bottleneck = EntropyBottleneck(self.compression_channels)

        if bottleneck_name is not None:
            self.bottleneck_name = bottleneck_name
        else:
            self.bottleneck_name = "bottleneck"

        if bottleneck_buffer_names is not None:
            self.bottleneck_buffer_names = bottleneck_buffer_names
        else:
            bottleneck_buffer_names = [
                "_cdf_length",
                "_offset",
                "_quantized_cdf",
            ]

            self.bottleneck_buffer_names = bottleneck_buffer_names

    def bottleneck_loss(self) -> Tensor:
        losses = []

        for module in self.modules():
            if isinstance(module, EntropyBottleneck):
                losses += [module.loss()]

        return Tensor(losses).sum()

    def update_bottleneck(self, force: bool = False) -> bool:
        updated = False

        for module in self.children():
            if not isinstance(module, EntropyBottleneck):
                continue

            updated |= module.update(force=force)

        return updated

    def load_state_dict(
        self, state_dict: OrderedDict[str, Tensor], strict: bool = True
    ):
        for bottleneck_buffer_name in self.bottleneck_buffer_names:
            name = f"{self.bottleneck_name}.{bottleneck_buffer_name}"

            size = state_dict[name].size()

            registered_buffers = []

            for name, buffer in self.bottleneck.named_buffers():
                if name == bottleneck_buffer_name:
                    registered_buffers += [buffer]

            if registered_buffers:
                registered_buffer = registered_buffers[0]

                if registered_buffer.numel() == 0:
                    registered_buffer.resize_(size)

        super(PriorAutoencoder, self).load_state_dict(state_dict, strict)

    def group_parameters(
        self,
    ) -> Tuple[Dict[str, Parameter], Dict[str, Parameter]]:
        parameters = {}

        for name, parameter in self.named_parameters():
            if not name.endswith(".quantiles") and parameter.requires_grad:
                parameters[name] = parameter

        bottleneck_parameters = {}

        for name, parameter in self.named_parameters():
            if name.endswith(".quantiles") and parameter.requires_grad:
                bottleneck_parameters[name] = parameter

        names = set(parameters.keys())

        bottleneck_names = set(bottleneck_parameters.keys())

        named_parameters = dict(self.named_parameters())

        assert len(names & bottleneck_names) == 0

        assert len(names | bottleneck_names) - len(set(named_parameters.keys())) == 0

        return parameters, bottleneck_parameters
