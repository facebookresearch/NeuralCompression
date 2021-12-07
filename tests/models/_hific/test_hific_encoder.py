# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from neuralcompression.models import HiFiCEncoder


class TestHiFiCEncoder:
    def test_forward(self):
        encoder = HiFiCEncoder((3, 256, 256))

        x = torch.rand((8, 3, 256, 256))

        assert encoder(x).shape == torch.Size([8, 220, 16, 16])
