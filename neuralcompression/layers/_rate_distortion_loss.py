"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
from typing import Dict, NamedTuple

import torch
from torch import Tensor
from torch.nn import Module, MSELoss


class _Outputs(NamedTuple):
    x_hat: Tensor
    scores: Dict[str, Tensor]


class _RateDistortionLosses(NamedTuple):
    bpp: float
    mse: float
    rate_distortion: float


class RateDistortionLoss(Module):
    def __init__(self, smoothness: float = 1e-2):
        super(RateDistortionLoss, self).__init__()

        self.mse = MSELoss()

        self.smoothness = smoothness

    def forward(self, outputs: _Outputs, target: Tensor) -> _RateDistortionLosses:
        n, _, h, w = target.size()

        bpps = []

        for scores in outputs.scores.values():
            bits = torch.log(scores).sum()

            pixels = -math.log(2) * (n * h * w)

            bpps += [float(bits / pixels)]

        bpp = sum(bpps)

        mse = self.mse(outputs.x_hat, target)

        rate_distortion = self.smoothness * 255 ** 2 * mse + bpp

        return _RateDistortionLosses(bpp, mse, rate_distortion)
