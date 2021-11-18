# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import NamedTuple, Optional

from compressai.entropy_models import EntropyBottleneck
from torch import Tensor

import neuralcompression.layers as layers
from ._prior import Prior


class _ForwardReturnTypeScores(NamedTuple):
    y: Tensor


class _ForwardReturnType(NamedTuple):
    scores: _ForwardReturnTypeScores
    x_hat: Tensor


class FactorizedPrior(Prior):
    def __init__(
        self,
        n: int = 128,
        m: int = 192,
        bottleneck_optimizer_lr: float = 1e-4,
        optimizer_lr: float = 1e-3,
        rate_distortion_smoothing: Optional[float] = None,
    ):
        super(FactorizedPrior, self).__init__(
            bottleneck_optimizer_lr,
            optimizer_lr,
        )

        self._architecture = layers.FactorizedPrior(n, m)

        self._rate_distortion_loss = layers.RateDistortionLoss(
            rate_distortion_smoothing,
        )

    def forward(self, x: Tensor) -> _ForwardReturnType:
        y_hat = self._architecture.encode(x)

        y_hat, y_scores = self._architecture._bottleneck_module(y_hat)

        x_hat = self._architecture.decode(y_hat)

        scores = _ForwardReturnTypeScores(y_scores)

        return _ForwardReturnType(scores, x_hat)

    def training_step(self, batch: Tensor, **kwargs):
        optimizer, auxiliary_optimizer = self.optimizers()

        optimizer.zero_grad()
        auxiliary_optimizer.zero_grad()

        outputs = self.forward(batch)

        rate_distortion_losses = self._rate_distortion_loss.forward(
            outputs.x_hat,
            dict(outputs.scores),
            batch,
        )

        optimizer.step()

        bottleneck_losses = []

        for module in self.modules():
            if isinstance(module, EntropyBottleneck):
                bottleneck_losses += [module.loss()]

        bottleneck_losses = sum(bottleneck_losses)

        auxiliary_optimizer.step()

        self.log_dict(
            {
                "training_bottleneck": bottleneck_losses,
                "training_bpp": rate_distortion_losses.bpp,
                "training_mse": rate_distortion_losses.mse,
                "training_rate_distortion": rate_distortion_losses.rate_distortion,
            },
            prog_bar=True,
        )
