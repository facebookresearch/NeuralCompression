# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from compressai.entropy_models import EntropyBottleneck

import neuralcompression.layers
from neuralcompression.layers import (
    AnalysisTransformation2D,
    SynthesisTransformation2D,
)
from neuralcompression.models import Prior


class TestPrior:
    autoencoder = neuralcompression.layers.Prior(
        AnalysisTransformation2D(128, 192),
        SynthesisTransformation2D(128, 192),
        EntropyBottleneck(192),
        "bottleneck",
        ["_cdf_length", "_offset", "_quantized_cdf"],
    )

    prior = Prior(autoencoder)

    def test__autoencoder_parameters(self):
        parameters = self.prior._autoencoder_parameters()

        assert len(parameters) == 54

    def test__bottleneck_optimizer_parameters(self):
        parameters = self.prior._bottleneck_optimizer_parameters()

        assert len(parameters) == 1

    def test__bottleneck_parameters(self):
        parameters = self.prior._bottleneck_parameters()

        assert len(parameters) == 1

    def test__intersection_parameters(self):
        parameters = self.prior._intersection_parameters()

        assert len(parameters) == 0

    def test__optimizer_parameters(self):
        parameters = self.prior._optimizer_parameters()

        assert len(parameters) == 54

    def test__parameters_dict(self):
        parameters = self.prior._parameters_dict()

        assert len(parameters) == 55

    def test__union_parameters(self):
        parameters = self.prior._union_parameters()

        assert len(parameters) == 55
