# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from compressai.entropy_models import EntropyBottleneck
from torch import Tensor

from neuralcompression.layers import (
    AnalysisTransformation2D,
    Prior,
    SynthesisTransformation2D,
)


class TestPrior:
    def test__bottleneck_loss(self):
        n, m = (128, 192)

        encoder_module = AnalysisTransformation2D(n, m)

        decoder_module = SynthesisTransformation2D(n, m)

        bottleneck_module = EntropyBottleneck(m)

        bottleneck_module_name = "bottleneck"

        bottleneck_buffer_names = [
            "_cdf_length",
            "_offset",
            "_quantized_cdf",
        ]

        prior = Prior(
            encoder_module,
            decoder_module,
            bottleneck_module,
            bottleneck_module_name,
            bottleneck_buffer_names,
        )

        assert isinstance(prior._bottleneck_loss, Tensor)

    def test_update(self):
        n, m = (128, 192)

        encoder_module = AnalysisTransformation2D(n, m)

        decoder_module = SynthesisTransformation2D(n, m)

        bottleneck_module = EntropyBottleneck(m)

        bottleneck_module_name = "bottleneck"

        bottleneck_buffer_names = [
            "_cdf_length",
            "_offset",
            "_quantized_cdf",
        ]

        prior = Prior(
            encoder_module,
            decoder_module,
            bottleneck_module,
            bottleneck_module_name,
            bottleneck_buffer_names,
        )

        updated = prior.update_bottleneck()

        assert updated
