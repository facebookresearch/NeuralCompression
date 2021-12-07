# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy.random
import scipy.special
import torch
import torch.testing

from neuralcompression.functional import log_ndtr


def test_log_ndtr():
    rng = numpy.random.default_rng(0xDEADBEEF)

    x = rng.random((32,), dtype=numpy.float)

    actual = log_ndtr(torch.tensor(x, dtype=torch.float))

    assert torch.isfinite(actual).all()

    torch.testing.assert_close(
        actual,
        torch.tensor(scipy.special.log_ndtr(x), dtype=torch.float),
    )
