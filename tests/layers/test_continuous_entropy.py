# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.testing
from torch import Size

from neuralcompression.distributions import NoisyNormal
from neuralcompression.layers import ContinuousEntropy


class TestContinuousEntropy:
    batch_shape = Size([16])

    prior = NoisyNormal(0.0, 1.0)

    prior._batch_shape = batch_shape

    continuous_entropy = ContinuousEntropy(
        prior=prior,
    )

    prior_shape = (32,)
    prior_dtype = torch.float32

    cdfs = torch.zeros((32,)).to(torch.int32)
    cdf_sizes = torch.zeros((32,)).to(torch.int32)
    cdf_offsets = torch.zeros((32,)).to(torch.int32)
    maximum_cdf_size = 16

    def test___init__(self):
        # missing `prior` or `prior_shape` and `prior_dtype`
        with pytest.raises(ValueError):
            ContinuousEntropy()

        # missing `prior` or `prior_dtype`
        with pytest.raises(ValueError):
            ContinuousEntropy(
                prior_shape=(32,),
            )

        # missing `prior` or `prior_shape`
        with pytest.raises(ValueError):
            ContinuousEntropy(
                prior_dtype=torch.int32,
            )

        # `prior` (no `prior_shape` and `prior_dtype`)
        continuous_entropy = ContinuousEntropy(
            prior=self.prior,
        )

        assert continuous_entropy._prior_shape == self.prior.batch_shape
        assert continuous_entropy._prior_dtype == self.prior_dtype

        # `prior` (no `prior_shape` and `prior_dtype`)
        continuous_entropy = ContinuousEntropy(
            prior_shape=self.prior_shape,
            prior_dtype=self.prior_dtype,
        )

        assert continuous_entropy._prior_shape == self.prior_shape
        assert continuous_entropy._prior_dtype == self.prior_dtype

        # `compressible` is `True`, `cdfs` is provided, but missing `cdf_sizes`
        # and `cdf_offsets`
        with pytest.raises(ValueError):
            ContinuousEntropy(
                compressible=True,
                prior=self.prior,
                cdfs=self.cdfs,
            )

        # `compressible` is `True`, `cdf_sizes` is provided, but missing `cdfs`
        # and `cdf_offsets`
        with pytest.raises(ValueError):
            ContinuousEntropy(
                compressible=True,
                prior=self.prior,
                cdf_sizes=self.cdf_sizes,
            )

        # `compressible` is `True`, `cdf_offsets` is provided, but missing
        # `cdfs` and `cdf_sizes`
        with pytest.raises(ValueError):
            ContinuousEntropy(
                compressible=True,
                prior=self.prior,
                cdf_offsets=self.cdf_offsets,
            )

        # `compressible` is `True`, but missing exactly one of `prior`, `cdfs`,
        # and `maximum_cdf_size`
        with pytest.raises(ValueError):
            ContinuousEntropy(
                compressible=True,
                prior_shape=self.prior_shape,
                prior_dtype=self.prior_dtype,
            )

            ContinuousEntropy(
                compressible=True,
                prior=self.prior,
                maximum_cdf_size=16,
            )

        # `compressible` is `True` and both `prior_shape` and `prior_dtype` are
        # provided, but `stateless` is `True` and `maximum_cdf_size` is
        # provided
        with pytest.raises(ValueError):
            ContinuousEntropy(
                compressible=True,
                stateless=True,
                prior_shape=self.prior_shape,
                prior_dtype=self.prior_dtype,
                maximum_cdf_size=self.maximum_cdf_size,
            )

        # `compressible` is `True`, both `prior_shape` and `prior_dtype` are
        # provided, and `maximum_cdf_size` is provided
        continuous_entropy = ContinuousEntropy(
            compressible=True,
            prior_shape=self.prior_shape,
            prior_dtype=self.prior_dtype,
            maximum_cdf_size=self.maximum_cdf_size,
        )

        torch.testing.assert_close(
            continuous_entropy._cdfs, torch.zeros((1, 16)).to(torch.int32)
        )

        torch.testing.assert_close(
            continuous_entropy._cdf_sizes, torch.zeros((16,)).to(torch.int32)
        )

        torch.testing.assert_close(
            continuous_entropy._cdf_offsets, torch.zeros((16,)).to(torch.int32)
        )

        # `compressible` is `True`, `prior` is provided
        continuous_entropy = ContinuousEntropy(
            compressible=True,
            prior=self.prior,
        )

        torch.testing.assert_close(
            continuous_entropy._cdfs.shape,
            Size([16, 6]),
        )

        torch.testing.assert_close(
            continuous_entropy._cdf_sizes.shape,
            Size([16]),
        )

        torch.testing.assert_close(
            continuous_entropy._cdf_offsets.shape,
            Size([16]),
        )

    def test_cdf_offsets(self):
        continuous_entropy = ContinuousEntropy(
            compressible=True,
            prior=self.prior,
        )

        assert continuous_entropy.cdf_offsets.shape == Size([16])

    def test_cdf_sizes(self):
        continuous_entropy = ContinuousEntropy(
            compressible=True,
            prior=self.prior,
        )

        assert continuous_entropy.cdf_sizes.shape == Size([16])

    def test_cdfs(self):
        continuous_entropy = ContinuousEntropy(
            compressible=True,
            prior=self.prior,
        )

        assert continuous_entropy.cdfs.shape == Size([16, 6])

    def test_coding_rank(self):
        assert self.continuous_entropy.coding_rank is None

    def test_compress(self):
        with pytest.raises(NotImplementedError):
            self.continuous_entropy.compress()

    def test_compressible(self):
        assert not self.continuous_entropy.compressible

    def test_context_shape(self):
        assert self.continuous_entropy.context_shape == Size([16])

    def test_decompress(self):
        with pytest.raises(NotImplementedError):
            self.continuous_entropy.decompress()

    def test_prior(self):
        assert True

    def test_prior_dtype(self):
        assert self.continuous_entropy.prior_dtype == torch.float32

    def test_prior_shape(self):
        assert self.continuous_entropy.prior_shape == Size([16])

    def test_quantize(self):
        assert True

    def test_range_coder_precision(self):
        assert self.continuous_entropy.range_coder_precision == 12

        continuous_entropy = ContinuousEntropy(
            prior=self.prior,
            range_coder_precision=16,
        )

        assert continuous_entropy.range_coder_precision == 16

    def test_stateless(self):
        assert not self.continuous_entropy.stateless

        continuous_entropy = ContinuousEntropy(
            prior=self.prior,
            stateless=True,
        )

        assert continuous_entropy.stateless

    def test_tail_mass(self):
        assert self.continuous_entropy.tail_mass == 0.00390625

        continuous_entropy = ContinuousEntropy(
            prior=self.prior,
            tail_mass=0.0,
        )

        assert continuous_entropy.tail_mass == 0.0
