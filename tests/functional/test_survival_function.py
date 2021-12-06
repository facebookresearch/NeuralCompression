# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy
import scipy.stats
import torch
import torch.testing
from torch.distributions import Normal

from neuralcompression.functional import survival_function


def test_survival_function():
    random = numpy.random.default_rng(0xDEADBEEF)

    batch_size = 32

    loc = torch.tensor([random.standard_normal(batch_size)])

    scale = torch.tensor([random.random(batch_size) + 1.0])

    x = torch.linspace(-8.0, 8.0, batch_size).to(torch.float64)

    torch.testing.assert_allclose(
        survival_function(x, Normal(loc, scale)),
        scipy.stats.norm(loc, scale).sf(x),
    )
