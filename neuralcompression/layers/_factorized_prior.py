"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import List, NamedTuple, OrderedDict

import torch
from torch import Size, Tensor

from ._analysis_transformation_2d import AnalysisTransformation2D
from ._prior import Prior
from ._synthesis_transformation_2d import SynthesisTransformation2D


class _CompressOutput(NamedTuple):
    compressed: List[List[str]]
    size: Size


class _DecompressOutput(NamedTuple):
    decompressed: Tensor


class _ForwardOutputScores(NamedTuple):
    y: Tensor


class _ForwardOutput(NamedTuple):
    scores: _ForwardOutputScores
    x_hat: Tensor


class FactorizedPrior(Prior):
    def __init__(self, n: int = 128, m: int = 192):
        super(FactorizedPrior, self).__init__(channels=m)

        self.encode = AnalysisTransformation2D(n, m)
        self.decode = SynthesisTransformation2D(n, m)

        self.n = n
        self.m = m

    def forward(self, x: Tensor) -> _ForwardOutput:
        encoded = self.encode(x)

        encoded, y_scores = self.bottleneck(encoded)

        scores = _ForwardOutputScores(y_scores)

        x_hat = self.decode(encoded)

        return _ForwardOutput(scores, x_hat)

    @classmethod
    def from_state_dict(cls, state_dict: OrderedDict[str, Tensor]):
        n = state_dict["encode.0.weight"].size()[0]
        m = state_dict["encode.6.weight"].size()[0]

        prior = cls(n, m)

        prior.load_state_dict(state_dict)

        return prior

    def compress(self, x: Tensor) -> _CompressOutput:
        y = self.encode(x)

        compressed = [self.bottleneck.compress(y)]

        size = Size(y.size()[-2:])

        return _CompressOutput(compressed, size)

    def decompress(
        self,
        compressed: List[List[str]],
        size: Size,
    ) -> _DecompressOutput:
        decoded = self.decode(self.bottleneck.decompress(compressed[0], size))

        decompressed = torch.clamp(decoded, 0, 1)

        return _DecompressOutput(decompressed)
