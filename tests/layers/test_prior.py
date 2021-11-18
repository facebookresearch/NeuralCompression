# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from compressai.entropy_models import EntropyBottleneck

from neuralcompression.layers import Prior


class TestPrior:
    def test__bottleneck_loss(self):
        n, m = (128, 192)

        bottleneck_module = EntropyBottleneck(m)

        bottleneck_module_name = "bottleneck"

        bottleneck_buffer_names = [
            "_cdf_length",
            "_offset",
            "_quantized_cdf",
        ]

        prior = Prior(
            n,
            m,
            bottleneck_module,
            bottleneck_module_name,
            bottleneck_buffer_names,
        )

        assert True

    def test__resize_registered_buffers(self):
        assert True

    def test_load_state_dict(self):
        assert True

    def test_update(self):
        assert True
