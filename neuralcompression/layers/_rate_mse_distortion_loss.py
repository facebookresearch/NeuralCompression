# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List, NamedTuple

import torch
from torch import Tensor
from torch.nn import Module, MSELoss


class RateDistortionLoss(NamedTuple):
    distortion: float
    rate: float
    rate_distortion: float


class RateMSEDistortionLoss(Module):
    """Rate-distortion loss.

    The rate-distortion loss is the minimum transmission bit-rate for a
    required quality. It can be obtained without consideration of a specific
    coding method. Rate is expressed in bits per pixel (BPP) of the original,
    ``x``, distortion is expressed as the mean squared error (MSE) between the
    original, ``x``, and the target, ``x_hat``.

    Args:
        trade_off: rate-distortion trade-off. `trade_off = 1` is the solution
            where the `(rate, distortion)` pair minimizes `rate + distortion`.
            Increasing `trade_off` will penalize the distortion term so more
            bits are spent.
        maximum: dynamic range of the input (i.e. the difference between the
            maximum the and minimum permitted values).
    """

    def __init__(self, trade_off: float = 1e-2, maximum: int = 255):
        super(RateMSEDistortionLoss, self).__init__()

        self._maximum = maximum

        self._trade_off = trade_off

        self._mse = MSELoss()

    def forward(
        self,
        x_hat: Tensor,
        probabilities: List[Tensor],
        x: Tensor,
    ) -> RateDistortionLoss:
        """
        Args:
            x_hat: encoder output.
            probabilities: reconstruction likelihoods.
            x: encoder input.
        """
        n, _, h, w = x.size()

        bpps = []

        for probability in probabilities:
            pixels = -math.log(2) * (n * h * w)

            bpps += [float(torch.log(probability).sum() / pixels)]

        rate = sum(bpps)

        distortion = self._mse(x_hat, x)

        rate_distortion = rate + distortion

        return RateDistortionLoss(
            rate,
            distortion,
            self._trade_off * self._maximum ** 2 * rate_distortion,
        )
