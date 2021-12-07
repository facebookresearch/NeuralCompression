# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy.random
import torch
import torch.testing

from neuralcompression.functional import log_expm1


def test_log_expm1():
    rng = numpy.random.default_rng(0xFEEEFEEE)

    x = rng.random((32,), dtype=numpy.float)

    actual = log_expm1(torch.tensor(x, dtype=torch.float))

    assert torch.isfinite(actual).all()

    torch.testing.assert_close(
        actual,
        torch.tensor(numpy.log(numpy.expm1(x)), dtype=torch.float),
    )
