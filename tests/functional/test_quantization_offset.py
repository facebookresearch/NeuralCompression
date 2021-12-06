# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch.distributions import Gamma, Laplace, Normal

from neuralcompression.functional import quantization_offset


def test_quantization_offset():
    assert quantization_offset(Gamma(5.0, 1.0)) == 0.0

    assert quantization_offset(Laplace(-2.0, 5.0)) == 0.0

    assert quantization_offset(Normal(3.0, 5.0)) == 0.0
