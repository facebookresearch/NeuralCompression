# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch.testing
from torch import Size
from torch.distributions import Normal

import neuralcompression.functional as ncF
from neuralcompression.distributions import UniformNoise


class TestUniformNoise:
    distribution = Normal(0.0, 1.0)

    generator = torch.Generator()

    generator.manual_seed(0xFEEEFEEE)

    x = torch.normal(0.0, 1.0, (32,), generator=generator)

    def test_cdf(self):
        torch.testing.assert_close(
            UniformNoise(self.distribution).cdf(self.x),
            self.distribution.cdf(self.x),
        )

    def test_entropy(self):
        torch.testing.assert_close(
            UniformNoise(self.distribution).entropy(),
            self.distribution.entropy(),
        )

    def test_enumerate_support(self):
        with pytest.raises(NotImplementedError):
            UniformNoise(self.distribution).enumerate_support()

    def test_expand(self):
        torch.testing.assert_close(
            UniformNoise(self.distribution).expand(Size([32])).batch_shape,
            Size([32]),
        )

    def test_icdf(self):
        torch.testing.assert_close(
            UniformNoise(self.distribution).icdf(self.x).shape,
            torch.Size([32]),
        )

    def test_log_cdf(self):
        torch.testing.assert_close(
            UniformNoise(self.distribution).log_cdf(self.x),
            ncF.log_cdf(self.x, self.distribution),
        )

    def test_log_prob(self):
        torch.testing.assert_close(
            UniformNoise(self.distribution).log_prob(self.x).shape,
            torch.Size([32]),
        )

    def test_log_survival_function(self):
        torch.testing.assert_close(
            UniformNoise(self.distribution).log_survival_function(self.x),
            ncF.log_survival_function(self.x, self.distribution),
        )

    def test_lower_tail(self):
        torch.testing.assert_close(
            UniformNoise(self.distribution).lower_tail(1.0),
            ncF.lower_tail(self.distribution, 1.0),
        )

    def test_mean(self):
        torch.testing.assert_close(
            UniformNoise(self.distribution).mean,
            torch.tensor(0.0),
        )

    def test_prob(self):
        torch.testing.assert_close(
            UniformNoise(self.distribution).prob(self.x).shape,
            torch.Size([32]),
        )

    def test_quantization_offset(self):
        torch.testing.assert_close(
            UniformNoise(self.distribution).quantization_offset,
            torch.tensor(0.0),
        )

    def test_rsample(self):
        torch.testing.assert_close(
            UniformNoise(self.distribution).rsample(torch.Size((5, 4))).shape,
            torch.Size([5, 4]),
        )

    def test_support(self):
        assert UniformNoise(self.distribution).support == self.distribution.support

    def test_survival_function(self):
        torch.testing.assert_close(
            UniformNoise(self.distribution).survival_function(self.x),
            ncF.survival_function(self.x, self.distribution),
        )

    def test_upper_tail(self):
        torch.testing.assert_close(
            UniformNoise(self.distribution).upper_tail(1.0),
            ncF.upper_tail(self.distribution, 1.0),
        )

    def test_variance(self):
        torch.testing.assert_close(
            UniformNoise(self.distribution).variance,
            torch.tensor(1.0),
        )
