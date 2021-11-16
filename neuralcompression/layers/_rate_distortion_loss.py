# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List, NamedTuple

import torch
from torch import Tensor
from torch.nn import Module, MSELoss


class _ForwardReturnType(NamedTuple):
    bpp: float
    mse: float
    rate_distortion: float


class RateDistortionLoss(Module):
    def __init__(self, _lambda: float = 1e-2):
        super(RateDistortionLoss, self).__init__()

        self._lambda = _lambda

        self._mse = MSELoss()

    def forward(
        self,
        x_hat: Tensor,
        scores: List[Tensor],
        x: Tensor,
    ) -> _ForwardReturnType:
        n, _, h, w = x.size()

        bpps = []

        for score in scores:
            bits = torch.log(score).sum()

            pixels = -math.log(2) * (n * h * w)

            bpps += [float(bits / pixels)]

        bpp = sum(bpps)

        mse = self._mse(x_hat, x)

        rate_distortion = self._lambda * 255 ** 2 * mse + bpp

        return _ForwardReturnType(bpp, mse, rate_distortion)
