import torch
from torch import Tensor
from torch.distributions import Distribution, Normal

from ._log_ndtr import log_ndtr


def _log_cdf(x: Tensor, distribution: Distribution) -> Tensor:
    if isinstance(distribution, Normal):
        standardized = (x - distribution.loc) / distribution.scale

        return log_ndtr(standardized)

    return torch.log(distribution.cdf(x))
