# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class IdentitySchedule(LambdaLR):
    """
    Identity learning rate schedule as a LambdaLR.

    Use this if you don't want your LambdaLR to change the learning rate.
    """

    def __init__(self, optimizer: Optimizer):
        def step_fn(step):
            return 1.0

        super().__init__(optimizer=optimizer, lr_lambda=step_fn)
