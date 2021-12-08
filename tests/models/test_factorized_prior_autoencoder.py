# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from neuralcompression.models import FactorizedPriorAutoencoder


class TestFactorizedPriorAutoencoder:
    def test_forward(self):
        factorized_prior = FactorizedPriorAutoencoder(128, 192)

        x = torch.rand(1, 3, 256, 256)

        x_hat, probabilities = factorized_prior.forward(x)

        assert x_hat.size() == x.size()

        assert probabilities

        y_probabilities = probabilities[0]

        assert y_probabilities.size()[0] == x.size()[0]
        assert y_probabilities.size()[1] == 192
        assert y_probabilities.size()[2] == x.size()[2] / 2 ** 4
        assert y_probabilities.size()[3] == x.size()[3] / 2 ** 4

    def test_from_state_dict(self, tmpdir):
        for n, m in [(128, 128), (128, 192), (192, 128)]:
            prior_a = FactorizedPriorAutoencoder(n, m)

            assert prior_a.network_channels == n
            assert prior_a.compression_channels == m

            path = tmpdir.join("model.pth.rar").strpath

            torch.save(prior_a.state_dict(), path)

            state_dict = torch.load(path)

            prior_b = FactorizedPriorAutoencoder.from_state_dict(state_dict)

            assert prior_a.network_channels == prior_b.network_channels
            assert prior_a.compression_channels == prior_b.compression_channels
