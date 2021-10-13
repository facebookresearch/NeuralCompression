import math

from torch import Tensor
from torch.distributions import Distribution

from ._estimate_tails import estimate_tails
from ._log_cdf import log_cdf


def lower_tail(distribution: Distribution, tail_mass: float) -> Tensor:
    """Approximates lower tail quantile for range coding.

    For range coding of random variables, the distribution tails need special
    handling, because range coding can only handle alphabets with a finite
    number of symbols. This method returns a cut-off location for the lower
    tail, such that approximately ``tail_mass`` probability mass is contained
    in the tails (together). The tails are then handled by using the ‘overflow’
    functionality of the range coder implementation (using a Golomb-like
    universal code).

    Args:
        distribution:
        tail_mass: desired probability mass for the tails.

    Returns:
        the approximate lower tail quantiles for each scalar distribution.
    """
    try:
        _lower_tail = distribution.icdf(tail_mass / 2)
    except (AttributeError, NotImplementedError):
        try:

            def _log_cdf(x: Tensor) -> Tensor:
                return log_cdf(x, distribution)

            _lower_tail = estimate_tails(
                _log_cdf,
                math.log(tail_mass / 2),
                distribution.batch_shape,
            )
        except NotImplementedError:
            error = "`distribution` must implement `cdf` or `icdf`"

            raise NotImplementedError(error)

    _lower_tail.requires_grad = False

    return _lower_tail
