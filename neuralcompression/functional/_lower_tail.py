import math

from torch import Tensor
from torch.distributions import Distribution

from ._estimate_tails import estimate_tails
from ._logcdf import logcdf


def lower_tail(distribution: Distribution, tail_mass: float) -> Tensor:
    try:
        _lower_tail = distribution.icdf(tail_mass / 2)
    except (AttributeError, NotImplementedError):
        try:

            def _logcdf(x: Tensor) -> Tensor:
                return logcdf(x, distribution)

            _lower_tail = estimate_tails(
                _logcdf,
                math.log(tail_mass / 2),
                distribution.batch_shape,
            )
        except NotImplementedError:
            error = "`distribution` must implement `cdf` or `icdf`"

            raise NotImplementedError(error)

    _lower_tail.requires_grad = False

    return _lower_tail
