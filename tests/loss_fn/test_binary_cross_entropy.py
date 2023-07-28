# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from neuralcompression.loss_fn import (
    BinaryCrossentropyDiscriminatorLoss,
    BinaryCrossentropyGeneratorLoss,
)


@pytest.fixture(params=[(3, 5), (3, 6, 6), (5, 3, 64, 64)])
def binary_logits(request):
    rng = torch.Generator()
    rng.manual_seed(int(torch.prod(torch.tensor(request.param))))

    logits1 = torch.randn(size=request.param, generator=rng)
    logits2 = torch.randn(size=request.param, generator=rng)
    target = torch.ones_like(logits1)

    return logits1, logits2, target


def test_binary_cross_entropy_disc(binary_logits):
    logits1, logits2, target = binary_logits

    loss_fn = BinaryCrossentropyDiscriminatorLoss()

    output = loss_fn(logits1, logits2, target)

    est1 = -1.0 * torch.log(torch.sigmoid(logits1)).mean()
    est2 = -1.0 * torch.log(1 - torch.sigmoid(logits2)).mean()

    assert torch.allclose(output, 0.5 * (est1 + est2))


def test_binary_cross_entropy_gen(binary_logits):
    logits1, _, target = binary_logits

    loss_fn = BinaryCrossentropyGeneratorLoss()

    output = loss_fn(logits1, target)

    est1 = -1.0 * torch.log(torch.sigmoid(logits1)).mean()

    assert torch.allclose(output, est1)
