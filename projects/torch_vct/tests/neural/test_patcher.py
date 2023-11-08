# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from projects.torch_vct.neural.patcher import Patcher


def test_pad():
    # create a Patcher instance with stride 2
    patcher = Patcher(stride=2)

    # create a simple input tensor
    x = torch.ones(2, 3, 4, 8)
    # pad the input tensor -- there should be no padding
    x_padded, n_patches_H, n_patches_W = patcher._pad(x, patch_size=2)
    assert x_padded.shape == (2, 3, 4, 8)
    # check that the number of patches in each dimension is correct
    assert n_patches_H == 2 and n_patches_W == 4

    patcher = Patcher(stride=3)
    x = torch.ones(2, 3, 5, 7)
    x_padded, n_patches_H, n_patches_W = patcher._pad(x, patch_size=5)
    assert x_padded.shape == (2, 3, 8, 11)
    assert n_patches_H == 2 and n_patches_W == 3

    # expect valueerror if number of pads to be added is not even, i.e
    # patch_size - stride is not even
    with pytest.raises(ValueError):
        x_padded, n_patches_H, n_patches_W = patcher._pad(x, patch_size=4)


def test_forward():
    patcher = Patcher(stride=3)
    x = torch.arange(2 * 3 * 5 * 7).view(2, 3, 5, 7).to(torch.float32)
    # apply the forward method to the input tensor
    patched = patcher(x, patch_size=5)
    assert patched.num_patches == (2, 3)
    # First dim is batch * total number of patches (2*3)
    # Second dim is seq len = patch_size * patch_size, last dim is C
    assert patched.tensor.shape == (2 * 2 * 3, 5 * 5, 3)

    # patch_size == stride, then unpached should equal padded input
    ps = 3
    patched = patcher(x, patch_size=ps)
    padded_input, *_ = patcher._pad(x, patch_size=ps)
    unpatched = patcher.unpatch(patched, channels_last=True)
    assert (
        padded_input.shape == unpatched.shape
    ), f"{unpatched.shape}, {padded_input.shape}"
    assert (unpatched == padded_input).all()
