# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy.random
import scipy.special
import torch
import torch.testing

from neuralcompression.functional import ndtr


def test_ndtr():
    torch.testing.assert_close(
        ndtr(torch.tensor([0.0])),
        torch.tensor([0.5]),
    )

    rng = numpy.random.default_rng(0xDEADBEEF)

    x = rng.random((32,), dtype=numpy.float)

    torch.testing.assert_close(
        ndtr(torch.tensor(x, dtype=torch.float)),
        torch.tensor(scipy.special.ndtr(x), dtype=torch.float),
    )
