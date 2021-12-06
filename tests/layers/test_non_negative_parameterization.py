# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.testing

from neuralcompression.layers import NonNegativeParameterization


class TestNonNegativeParameterization:
    def test___init__(self):
        x = torch.rand((1, 8, 8, 8)) * 2 - 1

        parameterization = NonNegativeParameterization(x)

        assert parameterization.initial_value.shape == x.shape

        assert torch.allclose(
            parameterization.initial_value,
            torch.sqrt(torch.max(x, x - x)),
            atol=2 ** -18,
        )

        for _ in range(10):
            minimum = torch.rand(1)

            x = torch.rand((1, 8, 8, 8)) * 2 - 1

            non_negative_parameterization = NonNegativeParameterization(
                minimum=minimum.item(),
                shape=x.shape,
            )

            reparameterized = non_negative_parameterization(x)

            assert reparameterized.shape == x.shape

            assert torch.allclose(reparameterized.min(), minimum)

    def test_forward(self):
        x = torch.rand((1, 8, 8, 8)) * 2 - 1

        parameterization = NonNegativeParameterization(
            shape=x.shape,
        )

        reparameterized = parameterization(x)

        assert reparameterized.shape == x.shape

        assert reparameterized.min() >= 0
