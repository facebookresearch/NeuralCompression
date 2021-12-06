# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Size

from neuralcompression.layers import HyperAnalysisTransformation2D


class TestHyperAnalysisTransformation:
    transformation = HyperAnalysisTransformation2D(28, 28)

    x = torch.rand((28, 28, 5, 5))

    assert transformation(x).shape == Size([28, 28, 2, 2])
