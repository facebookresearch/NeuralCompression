# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import random

import torch

from neuralcompression.functional import estimate_tails


def test_estimate_tails():
    def f(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.erf(x / math.sqrt(2)))

    target = 0.5 + random.randint(0, 50) / 100

    estimated = estimate_tails(f, target, 10)

    assert estimated.shape[0] == 10
