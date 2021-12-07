# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.testing

from neuralcompression.layers import GeneralizedDivisiveNormalization


class TestGeneralizedDivisiveNormalization:
    def test_backward(self):
        x = torch.rand((1, 32, 16, 16), requires_grad=True)

        normalization = GeneralizedDivisiveNormalization(32)

        normalized = normalization(x)

        normalized.backward(x)

        assert normalized.shape == x.shape

        assert x.grad is not None

        assert x.grad.shape == x.shape

        torch.testing.assert_allclose(
            x / torch.sqrt(1 + 0.1 * (x ** 2)),
            normalized,
        )

        normalization = GeneralizedDivisiveNormalization(
            32,
            inverse=True,
        )

        normalized = normalization(x)

        normalized.backward(x)

        assert normalized.shape == x.shape

        assert x.grad is not None

        assert x.grad.shape == x.shape

        torch.testing.assert_allclose(
            x * torch.sqrt(1 + 0.1 * (x ** 2)),
            normalized,
        )
