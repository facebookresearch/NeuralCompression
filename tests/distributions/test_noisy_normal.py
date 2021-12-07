# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch.distributions import Normal

from distributions.test_uniform_noise import TestUniformNoise


class TestNoisyNormal(TestUniformNoise):
    distribution = Normal(0.0, 1.0)
