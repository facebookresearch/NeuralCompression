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
        tail_mass: float = 2 ** (-8),
    ):
        super(ContinuousEntropy, self).__init__()

        self._distribution = distribution

        self._precision = precision

        self._tail_mass = tail_mass

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

        for index in range(data.size(0)):
            encoded = unbounded_index_range_encode(
                data,
                indexes,
                self._cdf,
                self._cdf_size,
                self._offset,
                self._precision,
                self._overflow_width,
            )

            compressed += [encoded]

        return compressed

    def decompress(
        self,
        data: Tensor,
        indexes: IntTensor,
        quantization_offsets: Optional[Tensor] = None,
        dtype: torch.dtype = torch.float,
        **kwargs,
    ):
        decompressed = self._cdf.new_empty(indexes.size())

        for index, encoded in enumerate(data):
            decoded = unbounded_index_range_decode(
                encoded,
                indexes[index],
                self._cdf,
                self._cdf_size,
                self._offset,
                self._precision,
                self._overflow_width,
            )

            decoded = torch.tensor(
                decoded,
                decompressed.dtype,
                decompressed.device,
            )

            decompressed[index] = decoded.reshape(decompressed[index].size())

        decompressed = self.reconstruct(
            decompressed,
            quantization_offsets,
            dtype,
        )

        return decompressed

    @staticmethod
    def quantize(
        x: Tensor,
        quantization_offsets: Optional[Tensor] = None,
    ) -> IntTensor:
        quantized = x.clone()

        if quantization_offsets:
            quantized -= quantization_offsets

        return torch.round(quantized).to(torch.int32)

    @staticmethod
    def reconstruct(
        x: Tensor,
        quantization_offsets: Optional[Tensor] = None,
        dtype: torch.dtype = torch.float,
    ):
        if quantization_offsets:
            return x.to(quantization_offsets.dtype) + quantization_offsets
        else:
            return x.to(dtype)
