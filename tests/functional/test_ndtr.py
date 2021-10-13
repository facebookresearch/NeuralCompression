"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.testing

from neuralcompression.functional import ndtr


def test_ndtr():
    torch.testing.assert_equal(
        ndtr(torch.tensor([0.0])),
        torch.tensor([0.5]),
    )

    torch.testing.assert_close(
        ndtr(torch.tensor([1.0])),
        torch.tensor([0.84134474606]),
    )
