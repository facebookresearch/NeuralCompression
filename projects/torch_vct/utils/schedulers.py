# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch


class LinearRampCosineLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Linear Ramp warm-up followed by cosine learning rate annealing
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_max: int,
        eta_min: float = 0,
        last_epoch: int = -1,
        ramp_len: int = 10,
    ):
        """Linear ramp warmup + cos annealing

        Args:
            optimizer: _description_
            T_max: max number of times scheduler is applied (~epochs)
            eta_min: min lr. Defaults to 0.
            last_epoch: last epoch. Defaults to -1.
            ramp_len: ramp length, should not be greater than T_max. Defaults to 10.
        """
        self.T_max = T_max
        self.eta_min = eta_min
        self.ramp_len = ramp_len
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        cosine_lr = [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * self.last_epoch / self.T_max))  # type: ignore
            / 2
            for base_lr in self.base_lrs  # type: ignore
        ]
        linear_lr = [
            base_lr * (1 + self.last_epoch) / self.ramp_len for base_lr in self.base_lrs  # type: ignore
        ]
        return [min(x, y) for x, y in zip(cosine_lr, linear_lr)]
