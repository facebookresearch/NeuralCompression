# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class RampConstCosineLRSchedule(LambdaLR):
    """
    Ramp-then-constant-then-cosine learning rate schedule.

    This class will first ramp the learning rate to a desired value, then have
    an extended constant portion, followed by a cosine learning rate decay.

    Args:
        optimizer: The optimize for which the learning rate should be modified.
        num_max_steps: The total steps for optimization.
        num_ramp_steps: The duration of the inital learning rate ramp.
        num_cosine_steps: The duration of cosine learning rate decay.
        min_val: A minimum value for the learning rate.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        num_max_steps: int,
        num_ramp_steps: int = 0,
        num_cosine_steps: int = 0,
        min_val: float = 1e-12,
    ):
        self._num_ramp_steps = num_ramp_steps
        self._num_cosine_steps = num_cosine_steps
        self._max_steps = num_max_steps
        self._min_val = min_val

        def step_fn(step):
            cosine_step = step - (self._max_steps - self._num_cosine_steps)
            if step < num_ramp_steps:
                return max(min(step / num_ramp_steps, 1.0), min_val)
            elif cosine_step > 0:
                angle = (cosine_step / self._num_cosine_steps) * math.pi / 2
                return max(math.cos(angle), min_val)
            else:
                return 1.0

        super().__init__(optimizer=optimizer, lr_lambda=step_fn)
