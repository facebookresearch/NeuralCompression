# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from neuralcompression.functional import soft_round, soft_round_inverse


def test_soft_round_inverse():
    x = torch.linspace(-2.0, 2.0, 50)

    torch.testing.assert_close(
        x,
        soft_round_inverse(x, alpha=1e-13),
    )

    x = torch.tensor([-1.25, -0.75, 0.75, 1.25])

    torch.testing.assert_close(
        x,
        soft_round_inverse(soft_round(x, alpha=2.0), alpha=2.0),
    )

    for offset in range(-5, 5):
        x = torch.linspace(offset + 0.001, offset + 0.999, 100)

        torch.testing.assert_close(
            torch.ceil(x) - 0.5,
            soft_round_inverse(x, alpha=5000.0),
            atol=0.001,
            rtol=0.002,
        )
