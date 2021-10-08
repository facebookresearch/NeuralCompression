"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from torch import tensor
from torch.testing import assert_close

from neuralcompression.functional import pmf_to_quantized_cdf


def test_pmf_to_quantized_cdf():
    assert_close(
        pmf_to_quantized_cdf(
            tensor([0.001, 0.01, 0.1, 1.0]),
            precision=8,
        ),
        tensor([0, 1, 3, 26, 256]),
    )
