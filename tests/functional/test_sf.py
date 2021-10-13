"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy
import scipy.stats
import torch
import torch.testing
from torch.distributions import Normal

from neuralcompression.functional import sf


def test_sf():
    random = numpy.random.default_rng(2021)

    batch_size = 50

    mu = torch.tensor([random.standard_normal(batch_size)])

    sigma = torch.tensor([random.random(batch_size) + 1.0])

    x = torch.linspace(-8.0, 8.0, batch_size).to(torch.float64)

    torch.testing.assert_allclose(
        sf(x, Normal(mu, sigma)),
        scipy.stats.norm(mu, sigma).sf(x),
    )
