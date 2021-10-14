import abc
from abc import ABCMeta
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.distributions import Distribution
from torch.nn import Module


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

    @property
    @abc.abstractmethod
    def probability_tables(self) -> Tuple[Tensor, Tensor, Tensor]:
        pass

    @abc.abstractmethod
    def compress(self, **kwargs):
        pass

    @abc.abstractmethod
    def decompress(self, **kwargs):
        pass

    @staticmethod
    def quantize(
        x: Tensor,
        quantization_offsets: Optional[Tensor] = None,
    ) -> Tensor:
        if quantization_offsets:
            x -= quantization_offsets

        x += torch.floor(x + 0.5) - x

        if quantization_offsets:
            return x + quantization_offsets

        return x

    @staticmethod
    def reconstruct(
        x: Tensor,
        quantization_offsets: Optional[Tensor] = None,
    ):
        if quantization_offsets:
            return x + quantization_offsets

        return x
