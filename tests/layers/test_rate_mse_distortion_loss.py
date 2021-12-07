# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from neuralcompression.layers import RateMSEDistortionLoss


class TestRateMSEDistortionLoss:
    def test_forward(self):
        x_hat = torch.ones(1, 32, 32, 3)

        scores = [
            torch.ones((32,)),
            torch.ones((32,)),
        ]

        x = torch.ones(1, 32, 32, 3)

        rate_distortion_loss = RateMSEDistortionLoss()

        rate_distortion_losses = rate_distortion_loss.forward(x_hat, scores, x)

        assert rate_distortion_losses.rate == 0.0
        assert rate_distortion_losses.distortion == 0.0
        assert rate_distortion_losses.rate_distortion == 0.0
