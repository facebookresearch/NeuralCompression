from typing import Tuple

from torch import Tensor

from ._continuous_entropy import ContinuousEntropy


class ContinuousIndexedEntropy(ContinuousEntropy):
    def __init__(self, **kwargs):
        super(ContinuousIndexedEntropy, self).__init__(**kwargs)

    @property
    def probabilities(self) -> Tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError
