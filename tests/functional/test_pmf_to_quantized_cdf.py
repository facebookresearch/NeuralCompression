"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.testing

from neuralcompression.functional import pmf_to_quantized_cdf


def test_pmf_to_quantized_cdf():
    torch.testing.assert_close(
        pmf_to_quantized_cdf(
            torch.tensor([0.001, 0.01, 0.1, 1.0]),
            precision=8,
        ),
        torch.tensor([0, 1, 3, 26, 256]),
    )
