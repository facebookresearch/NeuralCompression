import math

import torch
from torch import Tensor
from torch.distributions import Distribution

from ._estimate_tails import estimate_tails
from ._log_sf import log_sf


def upper_tail(distribution: Distribution, tail_mass: float) -> Tensor:
    try:
        _upper_tail = distribution.icdf(torch.tensor([1 - tail_mass / 2]))
    except (AttributeError, NotImplementedError):
        try:

            def _logsf(x: Tensor) -> Tensor:
                return log_sf(x, distribution)

            _upper_tail = estimate_tails(
                _logsf,
                math.log(tail_mass / 2),
                distribution.batch_shape,
            )
        except NotImplementedError:
            error = "`distribution` must implement `cdf` or `icdf`"

            raise NotImplementedError(error)

    _upper_tail.requires_grad = False

    return _upper_tail
