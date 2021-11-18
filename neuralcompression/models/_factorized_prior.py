# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from typing import Any, Dict, List, Tuple

from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor

import neuralcompression.layers as layers
from ._prior import Prior


class FactorizedPrior(Prior):
    def __init__(
        self,
        n: int = 128,
        m: int = 192,
        bottleneck_optimizer_lr: float = 1e-4,
        optimizer_lr: float = 1e-3,
        rate_distortion_trade_off: float = 1e-2,
        rate_distortion_maximum: int = 255,
    ):
        super(FactorizedPrior, self).__init__(
            layers.FactorizedPrior(n, m),
            bottleneck_optimizer_lr,
            optimizer_lr,
        )

        self.rate_distortion_loss = layers.RateMSEDistortionLoss(
            rate_distortion_trade_off,
            rate_distortion_maximum,
        )

    def forward(self, *args, **kwargs) -> Tuple[Tensor, List[Tensor]]:
        (x,) = args

        return self.architecture.forward(x)

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        batch: Tensor
        batch_idx: int
        optimizer_idx: int

        (
            batch,
            batch_idx,
            optimizer_idx,
        ) = args

        step_output: Dict[str, Any] = {}

        if optimizer_idx == 0:
            x_hat, probabilities = self.forward(batch)

            rate_distortion_losses = self.rate_distortion_loss.forward(
                x_hat,
                probabilities,
                batch,
            )

            losses = {
                "rate_loss": rate_distortion_losses.rate,
                "distortion_loss": rate_distortion_losses.distortion,
                "rate_distortion_loss": rate_distortion_losses.rate_distortion,
            }

            step_output = OrderedDict(
                {
                    "log": losses,
                    "loss": rate_distortion_losses.rate_distortion,
                    "progress_bar": losses,
                }
            )

        if optimizer_idx == 1:
            bottleneck_loss = self.architecture.bottleneck_loss

            losses = {
                "bottleneck_loss": bottleneck_loss,
            }

            step_output = OrderedDict(
                {
                    "log": losses,
                    "loss": bottleneck_loss,
                    "progress_bar": losses,
                }
            )

        return step_output
