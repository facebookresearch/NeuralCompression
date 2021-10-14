from typing import Tuple

import torch
from torch import Tensor

from ._continuous_entropy import ContinuousEntropy
from ...functional import unbounded_index_range_encode


class ContinuousIndexedEntropy(ContinuousEntropy):
    def __init__(self, **kwargs):
        super(ContinuousIndexedEntropy, self).__init__(**kwargs)

    @property
    def probability_tables(self) -> Tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError

    def compress(self, data: Tensor, index: Tensor, **kwargs) -> Tensor:
        return unbounded_index_range_encode(
            data,
            index,
            self._cdf,
            self._cdf_size,
            self._offset,
            self._precision,
            self._overflow_width,
        )

    def decompress(self, **kwargs):
        raise NotImplementedError
