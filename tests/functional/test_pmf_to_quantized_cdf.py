# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
        pmf_to_quantized_cdf(torch.tensor([1, 2, 3]), 8),
        torch.tensor([0, 42, 127, 256]),
    )
