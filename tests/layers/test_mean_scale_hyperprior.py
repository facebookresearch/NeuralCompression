"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch

from neuralcompression.layers import MeanScaleHyperprior


class TestMeanScaleHyperprior:
    def test_forward(self):
        mean_scale_hyperprior = MeanScaleHyperprior()

        x = torch.rand(1, 3, 64, 64)

        outputs = mean_scale_hyperprior.forward(x)

        assert hasattr(outputs, "scores")
        assert hasattr(outputs, "x_hat")

        assert hasattr(outputs.scores, "y")
        assert hasattr(outputs.scores, "z")

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
