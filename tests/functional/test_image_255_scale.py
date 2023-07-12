# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch import Tensor

import neuralcompression.functional as ncF


@pytest.mark.parametrize("dtype", [None, torch.float32, torch.long])
def test_image_255_scale(dtype: torch.dtype, arange_4d_image: Tensor):
    arange_4d_image = arange_4d_image.float()
    arange_4d_image = arange_4d_image / arange_4d_image.max()

    output = ncF.image_to_255_scale(arange_4d_image, dtype)

    if dtype is None:
        assert output.dtype == arange_4d_image.dtype
    else:
        assert output.dtype == dtype

    assert output.max() == 255
    assert output.min() == 0

    arange_4d_image = arange_4d_image / 2.0

    output = ncF.image_to_255_scale(arange_4d_image, dtype)

    assert output.max() == 128
    assert output.min() == 0

    arange_4d_image = arange_4d_image + 0.5

    output = ncF.image_to_255_scale(arange_4d_image, dtype)

    assert output.max() == 255
    assert output.min() == 128
