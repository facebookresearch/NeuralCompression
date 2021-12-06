# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from neuralcompression.models import ScaleHyperpriorAutoencoder


class TestScaleHyperpriorAutoencoder:
    def test_forward(self):
        scale_hyperprior = ScaleHyperpriorAutoencoder()

        scale_hyperprior.update_bottleneck()

        x = torch.rand(1, 3, 256, 256)

        x_hat, (y_scores, z_scores) = scale_hyperprior.forward(x)

        assert x_hat.size() == x.size()

        y_scores_size = y_scores.size()

        assert y_scores_size[0] == x.size()[0]
        assert y_scores_size[1] == 192
        assert y_scores_size[2] == x.size()[2] / 2 ** 4
        assert y_scores_size[3] == x.size()[3] / 2 ** 4

        z_scores_size = z_scores.size()

        assert z_scores_size[0] == x.size()[0]
        assert z_scores_size[1] == 128
        assert z_scores_size[2] == x.size()[2] / 2 ** 6
        assert z_scores_size[3] == x.size()[3] / 2 ** 6

    def test_from_state_dict(self, tmpdir):
        for n, m in [(128, 128), (128, 192), (192, 128)]:
            model = ScaleHyperpriorAutoencoder(n, m)

            filepath = tmpdir.join("model.pth.rar").strpath

            torch.save(model.state_dict(), filepath)

            loaded = ScaleHyperpriorAutoencoder.from_state_dict(torch.load(filepath))

            assert model.network_channels == loaded.network_channels
            assert model.compression_channels == loaded.compression_channels
