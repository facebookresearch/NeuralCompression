import math

import torch
from torch import Tensor
from torch.distributions import Distribution

from ._estimate_tails import estimate_tails
from ._log_sf import log_sf


def upper_tail(distribution: Distribution, tail_mass: float) -> Tensor:
    """Approximates upper tail quantile for range coding.

    For range coding of random variables, the distribution tails need special
        handling, because range coding can only handle alphabets with a finite
        number of symbols. This method returns a cut-off location for the upper
        tail, such that approximately tail_mass probability mass is contained
        in the tails (together). The tails are then handled by using the
        ‘overflow’ functionality of the range coder implementation (using a
        Golomb-like universal code).

    Args:
        distribution:
        tail_mass: desired probability mass for the tails.

    Returns:
        the approximate upper tail quantiles for each scalar distribution.
    """
    try:
        _upper_tail = distribution.icdf(torch.tensor([1 - tail_mass / 2]))
    except (AttributeError, NotImplementedError):
        try:
            def _log_sf(x: Tensor) -> Tensor:
                return log_sf(x, distribution)

            _upper_tail = estimate_tails(
                _log_sf,
                math.log(tail_mass / 2),
                distribution.batch_shape,
            )
        except NotImplementedError:
            error = "`distribution` must implement `cdf` or `icdf`"

            raise NotImplementedError(error)

    _upper_tail.requires_grad = False

    return _upper_tail
