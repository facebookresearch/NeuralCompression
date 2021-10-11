import abc
from abc import ABCMeta
from typing import Optional

import torch
from torch import Tensor
from torch.distributions import Distribution
from torch.nn import Module


class _ContinuousEntropy(Module, metaclass=ABCMeta):
    def __init__(
        self,
        distribution: Distribution,
        precision: int = 16,
        tail_mass: float = 2 ** (-8),
    ):
        super(_ContinuousEntropy, self).__init__()

        self.distribution = distribution

        self.tail_mass = tail_mass

        self.precision = precision

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
    def quantize(x: Tensor, offsets: Optional[Tensor] = None) -> Tensor:
        if offsets:
            x -= offsets

        x += torch.floor(x + 0.5) - x

        if offsets:
            return x + offsets

        return x

    @staticmethod
    def reconstruct(x: Tensor, offsets: Optional[Tensor] = None):
        if offsets:
            return x + offsets

        return x
