# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from torch import Tensor
from torch.distributions import Distribution

from ._estimate_tails import estimate_tails
from ._log_survival_function import log_survival_function


def upper_tail(distribution: Distribution, tail_mass: float) -> Tensor:
    """Approximates upper tail quantile for range coding.

    For range coding of random variables, the distribution tails need special
    handling, because range coding can only handle alphabets with a finite
    number of symbols. This method returns a cut-off location for the upper
    tail, such that approximately ``tail_mass`` probability mass is contained
    in the tails (together). The tails are then handled by using the overflow
    functionality of the range coder implementation (using a Golomb-like
    universal code).

    Args:
        distribution: an object representing a continuous-valued random
            variable.
        tail_mass: desired probability mass for the tails.

    Returns:
        the approximate upper tail quantiles for each scalar distribution.
    """
    try:
        x = distribution.icdf(torch.tensor([1 - tail_mass / 2]))
    except (AttributeError, NotImplementedError):
        try:
            x = estimate_tails(
                lambda _x: log_survival_function(_x, distribution),
                math.log(tail_mass / 2),
                distribution.batch_shape,
            )
        except NotImplementedError:
            error = """
            `distribution` must implement either `cdf` or `icdf`
            """

            raise NotImplementedError(error)

    return x.detach()
