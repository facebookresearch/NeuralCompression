# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.testing

from neuralcompression.functional import lower_bound


def test_lower_bound():
    x = torch.rand(16, requires_grad=True)

    bound = torch.rand(1)

    torch.testing.assert_allclose(
        lower_bound(x, bound),
        torch.max(x, bound),
    )

    bound = torch.rand(1)

    y = lower_bound(x, bound)

    torch.testing.assert_close(
        y,
        torch.max(x, bound),
    )

    y.backward(x)

    assert x.grad is not None

    torch.testing.assert_allclose(
        x.grad,
        (x >= bound) * x,
    )
