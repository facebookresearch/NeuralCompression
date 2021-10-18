"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pytest
import torch
import torch.testing

from neuralcompression.functional import pmf_to_quantized_cdf


def test_pmf_to_quantized_cdf():
    with pytest.raises(ValueError):
        pmf_to_quantized_cdf(torch.tensor([0.0]), 0)

    with pytest.raises(ValueError):
        pmf_to_quantized_cdf(torch.tensor([0.0]), 17)

    torch.testing.assert_close(
        pmf_to_quantized_cdf(torch.tensor([0.01, 0.1, 0.0, 1.0]), 16),
        torch.tensor([0, 589, 6493, 6494, 65536]),
    )
