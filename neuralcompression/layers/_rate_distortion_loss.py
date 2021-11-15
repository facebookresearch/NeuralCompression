# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, NamedTuple

import torch
from torch import Tensor
from torch.nn import Module, MSELoss


class _ForwardReturnType(NamedTuple):
    bpp: float
    mse: float
    rate_distortion: float


class RateDistortionLoss(Module):
    def __init__(self, smoothing: float = 1e-2):
        super(RateDistortionLoss, self).__init__()

        self.mse = MSELoss()

        self.smoothness = smoothing

    def forward(
        self,
        x_hat: Tensor,
        scores: Dict[str, Tensor],
        target: Tensor,
    ) -> _ForwardReturnType:
        n, _, h, w = target.size()

        bpps = []

        for scores in scores.values():
            bits = torch.log(scores).sum()

            pixels = -math.log(2) * (n * h * w)

            bpps += [float(bits / pixels)]

        bpp = sum(bpps)

        mse = self.mse(x_hat, target)

        rate_distortion = self.smoothness * 255 ** 2 * mse + bpp

        return _ForwardReturnType(bpp, mse, rate_distortion)
