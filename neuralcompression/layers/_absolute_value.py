# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor
from torch.nn import Module


class AbsoluteValue(Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.abs(x)
