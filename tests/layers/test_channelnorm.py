# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from torch import Tensor
import torch
from neuralcompression.layers import ChannelNorm2D


@pytest.mark.parametrize("affine", [True, False])
def test_channel_norm_2d(affine, arange_4d_image: Tensor):
    if arange_4d_image.shape[1] == 1:
        return
    norm = ChannelNorm2D(arange_4d_image.shape[1], affine=affine)

    mean = torch.mean(arange_4d_image, dim=1, keepdim=True)
    variance = torch.var(arange_4d_image, dim=1, keepdim=True)

    x_normed = (arange_4d_image - mean) * torch.rsqrt(variance + norm.epsilon)
    if affine is True:
        x_normed = norm.gamma * x_normed + norm.beta

    output = norm(arange_4d_image)

    assert torch.allclose(output, x_normed)
