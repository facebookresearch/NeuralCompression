# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import NamedTuple, Tuple

import torch
from torch import Tensor
from torch.nn import Module, MSELoss


class RateDistortionLoss(NamedTuple):
    distortion: Tensor
    rate: Tensor
    rate_distortion: Tensor


class RateMSEDistortionLoss(Module):
    """Rate-distortion loss.

    The rate-distortion loss is the minimum transmission bit-rate for a
    required quality. It can be obtained without consideration of a specific
    coding method. Rate is expressed in bits per pixel (BPP) of the original,
    :math:`x`, distortion is expressed as the mean squared error (MSE) between
    the original, :math:`x`, and the target, :math:`\\hat{x}`.

    Args:
        trade_off: rate-distortion trade-off. :math:`trade = 1` is the solution
            where :math:`(rate, distortion)` minimizes
            :math:`rate + distortion`. Increasing `trade_off` will penalize the
            distortion term so more bits are spent.
        maximum: dynamic range of the input (i.e. the difference between the
            maximum the and minimum permitted values).
    """

    def __init__(self, trade_off: float = 1e-2, maximum: int = 255):
        super(RateMSEDistortionLoss, self).__init__()

        self.maximum = maximum

        self.trade_off = trade_off

        self.mse = MSELoss()

    def forward(
        self,
        x_hat: Tensor,
        probabilities: Tuple[Tensor, ...],
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

        rate = torch.tensor(torch.sum(torch.tensor(bpps)), device=x.device)

        distortion = self.mse.forward(x_hat, x)

        rate_distortion = rate + distortion

        return RateDistortionLoss(
            rate,
            distortion,
            Tensor([self.trade_off * self.maximum ** 2 * rate_distortion]),
        )
