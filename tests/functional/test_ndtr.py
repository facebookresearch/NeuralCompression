"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy.random
import scipy.special
import torch
import torch.testing

from neuralcompression.functional import ndtr


def test_ndtr():
    rng = numpy.random.default_rng(2021)

    x = rng.random((28, 28))

    actual = ndtr(torch.tensor(x))

    assert torch.isfinite(actual).all()

    expected = torch.tensor(scipy.special.ndtr(x))

    torch.testing.assert_close(
        actual,
        expected,
    )
