# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn.functional as F

from neuralcompression.loss_fn import OASISDiscriminatorLoss, OASISGeneratorLoss


@pytest.fixture(params=[(3, 256, 4, 4), (20, 4, 12, 12), (5, 5, 64, 64)])
def oasis_logits(request):
    shape = request.param
    rng = torch.Generator()
    rng.manual_seed(int(torch.prod(torch.tensor(shape))))

    target_shape = (shape[0],) + shape[2:]

    logits1 = torch.randn(size=shape, generator=rng)
    logits2 = torch.randn(size=shape, generator=rng)
    target = torch.randint(low=0, high=shape[1] - 1, size=target_shape)

    return logits1, logits2, target


def test_binary_cross_entropy_disc(oasis_logits):
    logits1, logits2, target = oasis_logits

    loss_fn = OASISDiscriminatorLoss()

    output = loss_fn(logits1, logits2, target)

    est1 = torch.gather(
        -1.0 * F.log_softmax(logits1, dim=1), 1, target.unsqueeze(1) + 1
    ).mean()
    est2 = torch.gather(
        -1.0 * F.log_softmax(logits1, dim=1), 1, torch.zeros_like(target).unsqueeze(1)
    ).mean()

    # numerical precision issues
    assert torch.allclose(output, 0.5 * (est1 + est2), rtol=0.01)


def test_binary_cross_entropy_gen(oasis_logits):
    logits1, _, target = oasis_logits

    loss_fn = OASISGeneratorLoss()

    output = loss_fn(logits1, target)

    est1 = torch.gather(
        -1.0 * F.log_softmax(logits1, dim=1), 1, target.unsqueeze(1) + 1
    ).mean()

    # numerical precision issues
    assert torch.allclose(output, est1, rtol=0.01)
