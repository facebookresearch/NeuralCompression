# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, NamedTuple, OrderedDict, Tuple

import torch
from compressai.entropy_models import EntropyBottleneck
from torch import Size, Tensor

from ._analysis_transformation_2d import AnalysisTransformation2D
from ._prior import Prior
from ._synthesis_transformation_2d import SynthesisTransformation2D


class FactorizedPrior(Prior):
    def __init__(self, n: int = 128, m: int = 192):
        self._n = n
        self._m = m

        super(FactorizedPrior, self).__init__(
            AnalysisTransformation2D(self._n, self._m),
            SynthesisTransformation2D(self._n, self._m),
            EntropyBottleneck(self._m),
            "bottleneck",
            ["_cdf_length", "_offset", "_quantized_cdf"],
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        y = self.encoder.forward(x)

        y_hat, y_probabilities = self.bottleneck.forward(y)

        x_hat = self.decoder.forward(y_hat)

        return x_hat, [y_probabilities]

    @classmethod
    def from_state_dict(cls, state_dict: OrderedDict[str, Tensor]):
        n = state_dict["encoder.sequence.0.weight"].size()[0]
        m = state_dict["encoder.sequence.6.weight"].size()[0]

        prior = cls(n, m)

        prior.load_state_dict(state_dict)

        return prior

    def compress(self, x: Tensor) -> Tuple[List[List[str]], Size]:
        y = self.encoder.forward(x)

        return [self.bottleneck.compress(y)], Size(y.size()[-2:])

    def decompress(self, strings: List[List[str]], size: Size) -> Tensor:
        y_hat = self.bottleneck.decompress(strings[0], size)

        return torch.clamp(self.decoder(y_hat), 0, 1)
