# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from neuralcompression.functional import soft_round_conditional_mean


def test_soft_round_conditional_mean():
    for offset in range(-5, 5):
        x = torch.linspace(offset + 0.001, offset + 0.999, 100)

        torch.testing.assert_close(
            torch.round(x),
            soft_round_conditional_mean(x, alpha=5000.0),
            atol=0.001,
            rtol=0.001,
        )
