# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn


def _channel_norm_2d(input_channels, momentum=0.1, affine=True):
    return ChannelNorm2D(
        input_channels,
        momentum=momentum,
        affine=affine,
    )


class ChannelNorm2D(torch.nn.Module):
    def __init__(self, input_channels, momentum=0.1, epsilon=1e-3, affine=True):
        super(ChannelNorm2D, self).__init__()

        self.momentum = momentum

        self.epsilon = epsilon

        self.affine = affine

        if affine is True:
            self.gamma = torch.nn.Parameter(torch.ones(1, input_channels, 1, 1))
            self.beta = torch.nn.Parameter(torch.zeros(1, input_channels, 1, 1))

    def forward(self, x):
        mean = torch.mean(x, dim=1, keepdim=True)

        variance = torch.var(x, dim=1, keepdim=True)

        x_normed = (x - mean) * torch.rsqrt(variance + self.epsilon)

        if self.affine is True:
            x_normed = self.gamma * x_normed + self.beta

        return x_normed
