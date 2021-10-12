"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import scipy.special
import torch
import torch.testing

from neuralcompression.functional import ndtr


def test_ndtr():
    x = torch.rand((28, 28))

    actual = ndtr(torch.tensor(x))

    assert torch.isfinite(actual).all()

    expected = torch.tensor(scipy.special.ndtr(x.numpy()))

    torch.testing.assert_close(
        actual,
        expected,
    )
