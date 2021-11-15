# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from neuralcompression.layers import FactorizedPrior


class TestFactorizedPrior:
    def test_forward(self):
        factorized_prior = FactorizedPrior(128, 192)

        x = torch.rand(1, 3, 256, 256)

        outputs = factorized_prior.forward(x)

        assert hasattr(outputs, "scores")
        assert hasattr(outputs.scores, "y")
        assert hasattr(outputs, "x_hat")

        assert outputs.x_hat.shape == x.shape

        assert outputs.scores.y.size()[0] == x.size()[0]
        assert outputs.scores.y.size()[1] == 192
        assert outputs.scores.y.size()[2] == x.size()[2] / 2 ** 4
        assert outputs.scores.y.size()[3] == x.size()[3] / 2 ** 4
