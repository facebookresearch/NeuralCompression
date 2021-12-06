# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy
import scipy.stats
import torch
import torch.testing
from torch.distributions import Normal

from neuralcompression.functional import log_survival_function


def test_log_survival_function():
    rng = numpy.random.default_rng(0xFEEEFEEE)

    batch_size = 32

    loc = rng.standard_normal(batch_size, dtype=numpy.float)

    scale = rng.random(batch_size, dtype=numpy.float) + 1.0

    x = numpy.linspace(-8.0, 8.0, batch_size, dtype=numpy.float)

    torch.testing.assert_allclose(
        log_survival_function(
            torch.tensor(x, dtype=torch.float),
            Normal(
                torch.tensor(loc, dtype=torch.float),
                torch.tensor(scale, dtype=torch.float),
            ),
        ),
        scipy.stats.norm(loc, scale).logsf(x),
    )
