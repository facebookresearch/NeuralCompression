# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from neuralcompression.functional import soft_round


def test_soft_round():
    x = torch.linspace(-2.0, 2.0, 50)

    torch.testing.assert_close(
        x,
        soft_round(x, alpha=1e-13),
    )

    for offset in range(-5, 5):
        x = torch.linspace(offset - 0.499, offset + 0.499, 100)

        torch.testing.assert_close(
            torch.round(x),
            soft_round(x, alpha=2000.0),
            atol=0.02,
            rtol=0.02,
        )
