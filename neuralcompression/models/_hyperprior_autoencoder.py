# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck
from compressai.models import CompressionModel
from compressai.ops import quantize_ste
from torch import Tensor

from neuralcompression import HyperpriorCompressedOutput, HyperpriorOutput


class _HyperpriorAutoencoderBase(CompressionModel):
    """
    Base class for hyperprior autoencoder.

    This base class wraps a number of utility functions for working with
    Balle-style compression autoencoders implemented in the style of
    CompressAI, such as parameter collection and device management. Users are
    expected to subclass this calss before writing the appropriate PyTorch
    operations.
    """

    hyper_bottleneck: EntropyBottleneck
    _compress_cpu_layers: Optional[List[nn.Module]]

    def __init__(self):
        super().__init__()
        self._device_setting = "forward"
        self._compress_cpu_layers = None

    @property
    def device_setting(self) -> str:
        return self._device_setting

    def set_compress_cpu_layers(self, compress_cpu_layers: List[nn.Module]):
        self._compress_cpu_layers = compress_cpu_layers

    def collect_parameters(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
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
        params_dict = {
            k: p for k, p in self.named_parameters() if p.requires_grad is True
        }
        params_keys = set(params_dict.keys())

        inter_keys = model_keys.intersection(quantile_keys)
        union_keys = model_keys.union(quantile_keys)

        if len(inter_keys) != 0 or union_keys != params_keys:
            raise RuntimeError("Separating model and quantile parameters failed.")

        return [v for _, v in model_parameters.items()], [
            v for _, v in quantile_parameters.items()
        ]

    def _ste_quantize(self, latent: Tensor, means: Optional[Tensor] = None) -> Tensor:
        """
        Quantization with straight-through estimator.

        If the means are passed, then the input is first mean-shifted prior to
        quantization. Internally, the function uses the ``quantize_ste``
        function from CompressAI.

        Args:
            latent: The latent to quantize.
            means: Values to shift latent by prior to quantization.
        """
        if means is None:
            return quantize_ste(latent)
        else:
            return quantize_ste(latent - means) + means

    def _set_devices_for_compress(self):
        if self._compress_cpu_layers is None:
            raise RuntimeError("Must run set_compress_cpu_layers() in __init__()")

        for module in self._compress_cpu_layers:
            module.to(torch.device("cpu"))

    def _check_compress_devices(self) -> bool:
        if self._compress_cpu_layers is None:
            raise RuntimeError("Must run set_compress_cpu_layers() in __init__()")

        result = True
        cpu = torch.device("cpu")
        for module in self._compress_cpu_layers:
            for param in module.parameters():
                if not param.device == cpu:
                    result = False

        return result

    def update_tensor_devices(self, target_operation: str):
        raise NotImplementedError

    def _on_cpu(self):
        cpu = torch.device("cpu")
        for param in self.parameters():
            if param.device != cpu:
                return False
        return True

    def compress(
        self, image: Tensor, force_cpu: bool = True
    ) -> HyperpriorCompressedOutput:
        raise NotImplementedError

    def decompress(
        self, compressed_data: HyperpriorCompressedOutput, force_cpu: bool = True
    ) -> Tensor:
        raise NotImplementedError


class HyperpriorAutoencoderBase(_HyperpriorAutoencoderBase):
    def forward(self, image: Tensor) -> HyperpriorOutput:
        raise NotImplementedError


class ConditionalHyperpriorAutoencoderBase(_HyperpriorAutoencoderBase):
    def forward(self, image: Tensor, context) -> HyperpriorOutput:
        raise NotImplementedError
