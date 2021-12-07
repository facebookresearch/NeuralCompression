# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy.random
import scipy.special
import scipy.stats
import torch
import torch.testing
from torch.distributions import Normal, Uniform

from neuralcompression.functional import log_cdf


def test_log_cdf():
    rng = numpy.random.default_rng(0xFEEEFEEE)

    x = rng.random((32,), dtype=numpy.float)

    actual = log_cdf(torch.tensor(x, dtype=torch.float), Normal(0.0, 1.0))

    assert torch.isfinite(actual).all()

    torch.testing.assert_close(
        actual,
        torch.tensor(scipy.special.log_ndtr(x), dtype=torch.float),
    )

    actual = log_cdf(torch.tensor(x, dtype=torch.float), Uniform(0.0, 1.0))

    assert torch.isfinite(actual).all()

    torch.testing.assert_close(
        actual,
        torch.tensor(scipy.stats.uniform.logcdf(x), dtype=torch.float),
    )
