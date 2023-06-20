# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from torch import Tensor

import neuralcompression.functional as ncF


@pytest.mark.parametrize("factor", [64, 32, 16])
def test_pad_image(factor: int, arange_4d_image_odd: Tensor):
    output, (h, w) = ncF.pad_image_to_factor(arange_4d_image_odd, factor)

    assert h == arange_4d_image_odd.shape[2]
    assert w == arange_4d_image_odd.shape[3]

    if arange_4d_image_odd.shape[2] % factor != 0:
        est = (arange_4d_image_odd.shape[2] // factor + 1) * factor
        assert output.shape[2] == est

    if arange_4d_image_odd.shape[3] % factor != 0:
        est = (arange_4d_image_odd.shape[3] // factor + 1) * factor
        assert output.shape[3] == est
