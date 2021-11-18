# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, OrderedDict

import torch
import torch.nn.init
from compressai.entropy_models import EntropyBottleneck
from torch import Tensor
from torch.nn import (
    Conv2d,
    ConvTranspose2d,
    Module,
)


class Prior(Module):
    """A base class for implementing neural compression autoencoders.

    The class couples a ``bottleneck_module`` (e.g. the ``EntropyBottleneck``
    module provided by the CompressionAI package) with an autoencoder
    (i.e. ``encoder`` and ``decoder``).

    Using the base class is as straightforward as inheriting from the class and
    defining an ``encoder_module`` and ``decoder_module``. You may optionally
    provide a ``hyper_encoder_module`` and ``hyper_decoder_module`` (e.g. for
    implementing Hyperprior architectures).

    The ``neuralcompression.layers`` package includes a standard encoder
    (``AnalysisTransformation2D``), decoder (``SynthesisTransformation2D``),
    hyper encoder (``HyperAnalysisTransformation2D``), and hyper decoder
    (``HyperSynthesisTransformation2D``).

    Args:
        encoder_module:
        decoder_module:
        bottleneck_module:
        bottleneck_module_name:
        bottleneck_buffer_names:
        hyper_encoder_module:
        hyper_decoder_module:
    """

    def __init__(
        self,
        encoder_module: Module,
        decoder_module: Module,
        bottleneck_module: Module,
        bottleneck_module_name: str,
        bottleneck_buffer_names: List[str],
        hyper_encoder_module: Optional[Module] = None,
        hyper_decoder_module: Optional[Module] = None,
    ):
        super(Prior, self).__init__()

        self._encoder_module = encoder_module
        self._decoder_module = decoder_module

        self._bottleneck_module = bottleneck_module
        self._bottleneck_module_name = bottleneck_module_name
        self._bottleneck_buffer_names = bottleneck_buffer_names

        self._hyper_encoder_module = hyper_encoder_module
        self._hyper_decoder_module = hyper_decoder_module

        for module in self.modules():
            if isinstance(module, (Conv2d, ConvTranspose2d)):
                torch.nn.init.kaiming_normal_(module.weight)

                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    @property
    def _bottleneck_loss(self) -> float:
        losses = []

        for module in self.modules():
            if isinstance(module, EntropyBottleneck):
                losses += [module.loss()]

        return sum(losses)

    def update_bottleneck(self, force: bool = False) -> bool:
        updated = False

        for module in self.children():
            if not isinstance(module, EntropyBottleneck):
                continue

            updated |= module.update(force=force)

        return updated

    def load_state_dict(
        self,
        state_dict: OrderedDict[str, Tensor],
        strict: bool = True,
    ):
        """Copies parameters and buffers from ``state_dict`` into this module
        and its descendants. If strict is ``True``, then the keys of
        ``state_dict`` must exactly match the keys returned by this module’s
        ``torch.nn.Module.state_dict`` method.

        Args:
            state_dict: a ``dict`` containing parameters and persistent buffers.
            strict: whether to strictly enforce that the keys in ``state_dict``
                match the keys returned by this module’s
                ``torch.nn.Module.state_dict`` method, defaults to ``True``.
        """
        for bottleneck_buffer_name in self._bottleneck_buffer_names:
            name = f"{self._bottleneck_module_name}.{bottleneck_buffer_name}"

            size = state_dict[name].size()

            registered_buffers = []

            for name, buffer in self._bottleneck_module.named_buffers():
                if name == bottleneck_buffer_name:
                    registered_buffers += [buffer]

            if registered_buffers:
                registered_buffer = registered_buffers[0]

                if registered_buffer.numel() == 0:
                    registered_buffer.resize_(size)

        super(Prior, self).load_state_dict(state_dict, strict)
