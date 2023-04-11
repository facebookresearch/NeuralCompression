# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

## This module roughly corresponds to patcher.py in VCT

import math
from typing import NamedTuple, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Patched(NamedTuple):
    """Patched tensor

    Attributes:
        tensor: the patched tensor, expected shape is [B', C, patch_size*patch_size],
            with B'= B*num_patches_H*num_patches_W].
        num_patches: tuple (num_patches_H, num_patches_W), number of patches in B'.
    """

    tensor: Tensor
    num_patches: Tuple[int, int]


class Patcher(nn.Module):
    def __init__(self, stride: int, pad_mode: str = "reflect") -> None:
        """Pathching and unpatching tensors

        Args:
            stride: stride of the patcher
            pad_mode: how padding should be done. Defaults to "reflect".

        NOTE: this could potentially be implemented more efficiently with
            `torch.nn.functional.fold` and `unfold`
        """
        super().__init__()
        self.stride = stride
        self.pad_mode = pad_mode

    def _pad(self, x: Tensor, patch_size: int) -> Tuple[Tensor, int, int]:
        """Pad an input tensor x so that an integer number of patches of <patch_size>
            can be extracted from an input, i.e. we can do "valid" patch extraction

        Args:
            x: tensor to be padded; expected shape [B, C, H, W].
            patch_size: size of the patch.

        Returns: a tuple of 3 containing:
            - x_padded: tensor of shape [B, C, H`, W`] where sizes H` and W` are such
                that (size - patch_size + stride) % stride == 0
            - n_pathces_H: number of patches in the height dimension
            - n_patches_W: number of patches in the width dimension
        """
        assert x.dim() == 4, f"expected x to be of dim [B, C, H, W], got {x.shape}"
        if patch_size < self.stride:
            raise ValueError(
                f"<patch_size>={patch_size} must be greater than <stride>={self.stride}"
            )
        missing = patch_size - self.stride
        if missing % 2 != 0:
            raise ValueError(
                f"patch_size - self.stride={patch_size - self.stride} must be even!"
            )
        else:
            m = missing // 2

        H, W = x.shape[-2:]
        # H` and W` -- size of H and W after padding
        H_padded = math.ceil(H / self.stride) * self.stride
        W_padded = math.ceil(W / self.stride) * self.stride
        # number of patches in the height and width dimensions
        n_patches_H = H_padded // self.stride
        n_patches_W = W_padded // self.stride
        # need to specify padding for all dimensions explicitly: recall F.pad operates
        # backwards in dim, i.e. first pair refers to Width, second paid is H etc...
        pad_sizes = (m, W_padded - W + m, m, H_padded - H + m)
        return (
            F.pad(x, pad_sizes, mode=self.pad_mode),
            n_patches_H,
            n_patches_W,
        )

    def _pad_to_factor(self, x: Tensor, factor: int) -> Tensor:
        assert x.dim() == 4, f"expected x to be of dim [B, C, H, W], got {x.shape}"
        H, W = x.shape[-2:]
        H_padded = math.ceil(H / factor) * factor
        W_padded = math.ceil(W / factor) * factor
        pad_sizes = (0, W_padded - W, 0, H_padded - H)  # W then H! (pad is reverse)
        return F.pad(x, pad_sizes, mode=self.pad_mode)

    def _window_partition(self, x: Tensor, patch_size: int) -> Tensor:
        """Patchify without overlap

        Args:
            x: tensor of shape [B, C, H, W]
            patch_size: size of the patch

        Returns:
            Tensor of shape [B*num_patches_H*num_patches_W, patch_size^2, C]
        """
        # extract_patches_nonoverlapping + window_partition
        __pad = False
        B, C, H, W = x.shape
        if H % patch_size != 0 or W % patch_size != 0:
            if not __pad:
                raise ValueError(
                    f"input H, W={(H, W)} not divisible by patch size ({patch_size})"
                )
            x = self._pad_to_factor(x, factor=patch_size)
            _, H, W, _ = x.shape

        n_patches_H = H // patch_size
        n_patches_W = W // patch_size

        return (
            x.reshape(B, C, n_patches_H, patch_size, n_patches_W, patch_size)
            .permute(
                0, 2, 4, 3, 5, 1
            )  # [B, num_patches_H, num_patches_W, C, patch_size, patch_size]
            .contiguous()
            .reshape(B * n_patches_H * n_patches_W, patch_size**2, C)
        )

    def _window_partition_conv2d(self, x: Tensor, patch_size: int) -> Tensor:
        """Patchify with overlap

        Args:
            x: tensor of size [B, C, H, W]
            patch_size: patch size

        Returns:
            Tensor of patches with shape [B*num_patches_H*num_patches_W, patch_size^2, C]
        """
        B, C, H, W = x.shape

        # PyTorch expects [output C, input C, kernel H, kernel W]
        kernel = torch.diag(x.new_ones(patch_size**2 * C)).reshape(
            C * patch_size**2, C, patch_size, patch_size
        )
        patches = F.conv2d(
            x, kernel, stride=self.stride
        )  # [B, patch_size^2*C, num_patches_H, num_patches_W]
        n_patches_H, n_patches_W = patches.shape[-2:]
        return (
            patches.reshape(B, C, patch_size**2, n_patches_H, n_patches_W)
            .permute(0, 3, 4, 2, 1)  # [B, npatch_H, npatch_W, seq_len,  C]
            .contiguous()
            .reshape(B * n_patches_H * n_patches_W, patch_size**2, C)
        )

    def forward(self, x: Tensor, patch_size: int) -> Patched:
        """Pad and extract patches

        Args:
            x: tensor of shape [B, C, H, W]
            patch_size: patch size

        Returns:
            Patched object with two attributes:
                - `tensor` of shape [B*num_patches_H*num_patches_W, patch_size^2, C]
                - `num_patches` a tuple of ints (num_patches_W, num_patches_W) -- number
                    of patches in the height and width dimensions
        """
        x_padded, num_patches_H, num_patches_W = self._pad(x, patch_size)
        patches = (
            self._window_partition(x_padded, patch_size)
            if patch_size == self.stride
            else self._window_partition_conv2d(x_padded, patch_size)
        )  # [B*num_patches_H*num_patches_W, patch_size^2, C]

        return Patched(patches, (num_patches_H, num_patches_W))

    def unpatch(
        self,
        x_patched: Union[Patched, Tuple[Tensor, Tuple[int, int]]],
        crop: Optional[Tuple[int, int]] = None,
        channels_last: bool = False,
    ) -> Tensor:
        """Unpatch a patched object.

        This method only applies when stride and patch_size are the same.

        Args:
            x_patched: patched object or a tuple (tensor, (patches_h, patches_w))
            crop: crop the height and width of the ptches
            channels_last: are the channels in the last dimension? Defaults to False

        Returns:
            Tensor of shape [B, C, H, W]
        """
        x, (n_patches_H, n_patches_W) = x_patched

        if not channels_last:
            x = x.permute(0, 2, 1).contiguous()

        _, seq_len, C = x.shape

        # Can only unpatch objects where stride == patch_size
        assert (
            seq_len == self.stride**2
        ), "Length mismatch. Potential reason: unpatch can only handle patch_size=stride"
        x = (
            x.reshape(-1, n_patches_H, n_patches_W, self.stride, self.stride, C)
            .permute(0, 5, 1, 3, 2, 4)
            .contiguous()
            .reshape(-1, C, n_patches_H * self.stride, n_patches_W * self.stride)
        )

        return x[..., : crop[0], : crop[1]] if crop is not None else x
