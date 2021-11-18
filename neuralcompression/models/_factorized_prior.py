# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import OrderedDict
from typing import NamedTuple, Optional, Tuple, List, Any, Dict

from compressai.entropy_models import EntropyBottleneck
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.utilities.types import STEP_OUTPUT
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
        rate_distortion_trade_off: float = 1e-2,
        rate_distortion_maximum: int = 255,
    ):
        self._autoencoder = layers.FactorizedPrior(n, m)

        super(FactorizedPrior, self).__init__(
            self._autoencoder,
            bottleneck_optimizer_lr,
            optimizer_lr,
        )

        self._rate_distortion_loss = layers.RateMSEDistortionLoss(
            rate_distortion_trade_off,
            rate_distortion_maximum,
        )

    def forward(self, *args, **kwargs) -> Tuple[Tensor, List[Tensor]]:
        (x,) = args

        y_hat = self._autoencoder._encoder_module(x)

        y_hat, y_probabilities = self._autoencoder._bottleneck_module.forward(y_hat)

        x_hat = self._autoencoder._decoder_module(y_hat)

        return x_hat, [y_probabilities]

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

            rate_distortion_losses = self._rate_distortion_loss.forward(
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
            bottleneck_loss = self._autoencoder._bottleneck_loss

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
