from typing import Dict, Optional

from torch import Size, Tensor
from torch.distributions import Distribution
from torch.distributions.constraints import Constraint


class Factorized(Distribution):
    def __init__(self, **kwargs):

        super(Factorized, self).__init__(**kwargs)

    @property
    def arg_constraints(self) -> Dict[str, Constraint]:
        raise NotImplementedError

    @property
    def mean(self) -> Tensor:
        raise NotImplementedError

    @property
    def support(self) -> Optional[Constraint]:
        raise NotImplementedError

    @property
    def variance(self) -> Tensor:
        raise NotImplementedError

    def cdf(self, value: Tensor) -> Tensor:
        raise NotImplementedError

    def entropy(self) -> Tensor:
        raise NotImplementedError

    def enumerate_support(self, expand: bool = True) -> Tensor:
        raise NotImplementedError

    def expand(
        self,
        batch_shape: Size,
        _instance: Optional[Distribution] = None,
    ) -> Distribution:
        raise NotImplementedError

    def icdf(self, value: Tensor) -> Tensor:
        raise NotImplementedError

    def log_prob(self, value: Tensor) -> Tensor:
        raise NotImplementedError

    def rsample(self, sample_shape: Size = Size()) -> Tensor:
        raise NotImplementedError
