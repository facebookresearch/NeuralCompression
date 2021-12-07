# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from neuralcompression.models import HiFiCDiscriminator


class TestHiFiCDiscriminator:
    def test_forward(self):
        discriminator = HiFiCDiscriminator()

        x = torch.rand((8, 3, 256, 256))
        y = torch.rand((8, 220, 16, 16))

        a, b = discriminator(x, y)

        assert a.shape == torch.Size([2048, 1])
        assert b.shape == torch.Size([2048, 1])
