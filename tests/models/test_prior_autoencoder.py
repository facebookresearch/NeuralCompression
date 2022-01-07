# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from compressai.entropy_models import EntropyBottleneck
from torch import Tensor

from neuralcompression.layers import AnalysisTransformation2D, SynthesisTransformation2D
from neuralcompression.models import PriorAutoencoder


class TestPriorAutoencoder:
    def test_bottleneck_loss(self):
        n, m = (128, 192)

        prior = PriorAutoencoder(
            n,
            m,
            3,
            AnalysisTransformation2D(n, m),
            SynthesisTransformation2D(n, m),
            EntropyBottleneck(m),
            "bottleneck",
            [
                "_cdf_length",
                "_offset",
                "_quantized_cdf",
            ],
        )

        assert isinstance(prior.bottleneck_loss(), Tensor)

    def test_update(self):
        n, m = (128, 192)

        prior = PriorAutoencoder(
            n,
            m,
            3,
            AnalysisTransformation2D(n, m),
            SynthesisTransformation2D(n, m),
            EntropyBottleneck(m),
            "bottleneck",
            [
                "_cdf_length",
                "_offset",
                "_quantized_cdf",
            ],
        )

        updated = prior.update_bottleneck()

        assert updated
