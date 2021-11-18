# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, NamedTuple, OrderedDict, Tuple

import torch
from torch import Size, Tensor

from ._analysis_transformation_2d import AnalysisTransformation2D
from ._prior import Prior
from ._synthesis_transformation_2d import SynthesisTransformation2D


class _ForwardOutputScores(NamedTuple):
    y: Tensor


class _ForwardOutput(NamedTuple):
    scores: _ForwardOutputScores
    x_hat: Tensor


class FactorizedPrior(Prior):
    def __init__(self, n: int = 128, m: int = 192):
        super(FactorizedPrior, self).__init__(n, m)

        self.encode = AnalysisTransformation2D(n, m)
        self.decode = SynthesisTransformation2D(n, m)

        self.n = n
        self.m = m

    def forward(self, x: Tensor) -> _ForwardOutput:
        y = self.encode(x)

        y_hat, y_probabilities = self.bottleneck(y)

        probabilities = _ForwardOutputScores(y_probabilities)

        x_hat = self.decode(y_hat)

        return _ForwardOutput(probabilities, x_hat)

    @classmethod
    def from_state_dict(cls, state_dict: OrderedDict[str, Tensor]):
        n = state_dict["encode.0.weight"].size()[0]
        m = state_dict["encode.6.weight"].size()[0]

        prior = cls(n, m)

        prior.load_state_dict(state_dict)

        return prior

    def compress(self, x: Tensor) -> Tuple[List[List[str]], Size]:
        y = self.encode(x)

        return [self.bottleneck.compress(y)], Size(y.size()[-2:])

    def decompress(self, compressed: List[List[str]], size: Size) -> Tensor:
        y_hat = self.bottleneck.decompress(compressed[0], size)

        return torch.clamp(self.decode(y_hat), 0, 1)
