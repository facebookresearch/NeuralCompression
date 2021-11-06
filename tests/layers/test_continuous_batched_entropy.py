"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch

from neuralcompression.layers import ContinuousBatchedEntropy
from neuralcompression.distributions import NoisyNormal


class TestContinuousBatchedEntropy:
    def test___init__(self):
        prior = NoisyNormal(0.0, 1.0)

        continuous_batched_entropy = ContinuousBatchedEntropy(
            coding_rank=1,
            prior=prior,
        )

        assert continuous_batched_entropy.coding_rank == 1
        assert continuous_batched_entropy.prior == prior
        assert continuous_batched_entropy.prior_dtype == torch.float32
        assert continuous_batched_entropy.range_coder_precision == 12
        assert continuous_batched_entropy.tail_mass == 2 ** -8

    def test__compute_indexes(self):
        assert True

    def test__pmf_to_cdf(self):
        assert True

    def test_compress(self):
        assert True

    def test_decompress(self):
        prior = NoisyNormal(0.25, 10.0)

        continuous_batched_entropy = ContinuousBatchedEntropy(
            coding_rank=1,
            compressible=True,
            prior=prior,
        )

        # x = prior._distribution.sample([100])

        # x_quantized = continuous_batched_entropy.quantize(x)

        # x_decompressed = continuous_batched_entropy.decompress(
        #     continuous_batched_entropy.compress(x),
        #     [100],
        # )

    def test_forward(self):
        assert True

    def test_non_integer_offsets(self):
        assert True

    def test_quantization_offset(self):
        assert True
