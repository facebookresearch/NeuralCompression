# Copyright 2020 - present, Facebook, Inc
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
#
# Modifications
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class XCA(nn.Module):
    """
    Cross-Covariance Attention (XCA).

    This is a redistirbution of cross-covariance implementation originally
    released at https://github.com/facebookresearch/xcit.

    Cross-covariance attention was introduced in the following paper:

    XCiT: Cross-Covariance Image Transformers
    A El-Nouby, H Touvron, M Caron, P Bojanowski, M Douze, A Joulin, I Laptev,
    N Neverova, G Synnaeve, J Verbeek, H Jegou

    Operation where the channels are updated using a weighted sum. The weights
    are obtained from the (softmax normalized) Cross-covariance matrix
    (Q^T \\cdot K \\in d_h \\times d_h)/

    Args:
        dim: Dimension of input.
        num_heads: Number of attention heads.
        qkv_bias: Whether to use bias in qkv term.
        attn_drop: Attention dropout probability.
        proj_drop: Projection dropout probability.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        # Result of next line is (qkv, B, num (H)eads,  (C')hannels per head, N)
        qkv: Tensor = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 4, 1)
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # Paper section 3.2 l2-Normalization and temperature scaling
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # (B, H, C', N), permute -> (B, N, H, C')
        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"temperature"}
