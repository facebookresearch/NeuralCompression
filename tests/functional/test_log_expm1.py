"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.testing

from neuralcompression.functional import log_expm1


def test_log_expm1():
    assert torch.isfinite(log_expm1(torch.tensor(0.00001)))

    assert torch.isfinite(log_expm1(torch.tensor(10000.0)))
