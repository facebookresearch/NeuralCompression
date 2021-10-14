"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import abc
from abc import ABCMeta
from typing import Optional, Tuple

import torch
import torch.nn.functional
from torch import Tensor, IntTensor
from torch.distributions import Distribution
from torch.nn import Module

from ...functional import (
    unbounded_index_range_encode,
    unbounded_index_range_decode,
)


class ContinuousEntropy(Module, metaclass=ABCMeta):
    _overflow_width: int = 4

    def __init__(
        self,
        distribution: Distribution,
        precision: int = 16,
    ):
        super(ContinuousEntropy, self).__init__()

        self._distribution = distribution

        self._precision = precision

        self.register_buffer("_cdf", IntTensor())
        self.register_buffer("_cdf_size", IntTensor())
        self.register_buffer("_offset", IntTensor())

    @property
    @abc.abstractmethod
    def probabilities(self) -> Tuple[Tensor, Tensor, Tensor]:
        pass

    def compress(
        self,
        data: Tensor,
        indexes: IntTensor,
        quantization_offsets: Optional[Tensor] = None,
    ):
        data = self.quantize(data, quantization_offsets)

        compressed = []

        for index in range(data.size()):
            compressed += [(unbounded_index_range_encode(
                data,
                indexes,
                self._cdf,
                self._cdf_size,
                self._offset,
                self._precision,
                self._overflow_width,
            ))]

        return compressed

    def decompress(
        self,
        data: Tensor,
        indexes: IntTensor,
        quantization_offsets: Optional[Tensor] = None,
        dtype: torch.dtype = torch.float,
    ):
        decompressed = self._cdf.new_empty(indexes.size())

        for index, _ in enumerate(data):
            decompressed[index] = torch.tensor(
                unbounded_index_range_decode(
                    data[index],
                    indexes[index],
                    self._cdf,
                    self._cdf_size,
                    self._offset,
                    self._precision,
                    self._overflow_width,
                ),
                decompressed.dtype,
                decompressed.device,
            ).reshape(decompressed[index].size())

        return self.reconstruct(
            decompressed,
            quantization_offsets,
            dtype,
        )

    @staticmethod
    def quantize(
        x: Tensor,
        quantization_offsets: Optional[Tensor] = None,
    ) -> IntTensor:
        x = x.clone()

        if quantization_offsets:
            x -= quantization_offsets

        return torch.round(x).to(torch.int32)

    @staticmethod
    def reconstruct(
        x: Tensor,
        quantization_offsets: Optional[Tensor] = None,
        dtype: torch.dtype = torch.float,
    ) -> Tensor:
        if quantization_offsets:
            return x.to(quantization_offsets.dtype) + quantization_offsets
        else:
            return x.to(dtype)
