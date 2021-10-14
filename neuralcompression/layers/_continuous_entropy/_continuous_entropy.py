import abc
from abc import ABCMeta
from typing import Optional

import torch
from torch import Tensor
from torch.distributions import Distribution
from torch.nn import Module


class ContinuousEntropy(Module, metaclass=ABCMeta):
    def __init__(
        self,
        distribution: Distribution,
        precision: int = 16,
        tail_mass: float = 2 ** (-8),
    ):
        super(ContinuousEntropy, self).__init__()

        self.distribution = distribution

        self.precision = precision

        self.tail_mass = tail_mass

    @property
    @abc.abstractmethod
    def probability_tables(self):
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
