# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Size

from neuralcompression.layers import AnalysisTransformation2D


class TestAnalysisTransformation2D:
    transformation = AnalysisTransformation2D(28, 28)

    x = torch.rand((16, 3, 28, 28))

    assert transformation(x).shape == Size([16, 28, 2, 2])
