"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy.random
import scipy.special
import torch
import torch.testing
from torch.distributions import Normal

from neuralcompression.functional import log_cdf


def test_log_ndtr():
    rng = numpy.random.default_rng(2021)

    x = rng.random((28, 28))

    distribution = Normal(torch.tensor([0.0]), torch.tensor([1.0]))

    actual = log_cdf(torch.tensor(x), distribution)

    assert torch.isfinite(actual).all()

    expected = torch.tensor(scipy.special.log_ndtr(x))

    torch.testing.assert_close(
        actual,
        expected,
    )
