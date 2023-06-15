# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch import Tensor


class ChannelNorm2D(nn.Module):
    """
    Channel normalization layer.

    This implements the channel normalization layer as described in the
    following paper:

    High-Fidelity Generative Image Compression
    F. Mentzer, G. Toderici, M. Tschannen, E. Agustsson

    Using this layer provides more stability to model outputs when there is a
    shift in image resolutions between the training set and the test set.

    Args:
        input_channels: Number of channels to normalize.
        epsilon: Divide-by-0 protection parameter.
        affine: Whether to include affine parameters for the noramlized output.
    """

    def __init__(self, input_channels: int, epsilon: float = 1e-3, affine: bool = True):
        super().__init__()

        if input_channels <= 1:
            raise ValueError(
                "ChannelNorm only valid for channel counts greater than 1."
            )

        self.epsilon = epsilon
        self.affine = affine

        if affine is True:
            self.gamma = nn.Parameter(torch.ones(1, input_channels, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, input_channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        mean = torch.mean(x, dim=1, keepdim=True)
        variance = torch.var(x, dim=1, keepdim=True)

        x_normed = (x - mean) * torch.rsqrt(variance + self.epsilon)

        if self.affine is True:
            x_normed = self.gamma * x_normed + self.beta

        return x_normed
