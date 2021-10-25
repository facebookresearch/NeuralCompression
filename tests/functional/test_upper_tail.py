"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.testing
from torch.distributions import Normal, Uniform

from neuralcompression.functional import upper_tail


def test_upper_tail():
    torch.testing.assert_close(
        upper_tail(
            Normal(0.0, 1.0),
            0.5,
        ),
        torch.tensor([0.6745]),
    )

    torch.testing.assert_close(
        upper_tail(
            Uniform(0.0, 1.0),
            0.5,
        ),
        torch.tensor([0.75]),
    )
