"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch

from neuralcompression.layers import ScaleHyperprior


class TestScaleHyperprior:
    def test_forward(self):
        scale_hyperprior = ScaleHyperprior()

        x = torch.rand(1, 3, 64, 64)

        outputs = scale_hyperprior.forward(x)

        assert hasattr(outputs, "scores")
        assert hasattr(outputs.scores, "y")
        assert hasattr(outputs.scores, "z")
        assert hasattr(outputs, "x_hat")

        assert outputs.x_hat.size() == x.size()

        y_scores_size = outputs.scores.y.size()

        assert y_scores_size[0] == x.size()[0]
        assert y_scores_size[1] == 192
        assert y_scores_size[2] == x.size()[2] / 2 ** 4
        assert y_scores_size[3] == x.size()[3] / 2 ** 4

        z_scores_size = outputs.scores.z.size()

        assert z_scores_size[0] == x.size()[0]
        assert z_scores_size[1] == 128
        assert z_scores_size[2] == x.size()[2] / 2 ** 6
        assert z_scores_size[3] == x.size()[3] / 2 ** 6

    def test_from_state_dict(self, tmpdir):
        for n, m in [(128, 128), (128, 192), (192, 128)]:
            model = ScaleHyperprior(n, m)

            filepath = tmpdir.join("model.pth.rar").strpath

            torch.save(model.state_dict(), filepath)

            loaded = ScaleHyperprior.from_state_dict(torch.load(filepath))

            assert model._n == loaded._n
            assert model._m == loaded._m
