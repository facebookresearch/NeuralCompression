# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, List, Optional, Tuple

import scipy.stats as sps
import torch
import torch.nn as nn
from compressai.entropy_models import EntropyModel
from compressai.ops import LowerBound
from torch import Tensor


class Scaler:
    """Helper to scale `scales` before quantizing if to integer indices.

    Log quantization grid that aligns woth how scales are distributed
    """

    def __init__(
        self,
        scales_min: float = 0.01,
        scales_max: float = 256.0,
        num_bins: int = 256,
        verify_valid_scales: bool = True,
    ) -> None:
        self.scales_min = float(scales_min)
        self.scales_max = float(scales_max)
        self.num_bins = int(num_bins)
        self.verify_valid_scales = bool(verify_valid_scales)

    def to_scale_idx(self, scales: Tensor, training: bool = True) -> Tensor:
        """Convert `scale` in [0, scales_max] to an index."""
        if self.verify_valid_scales:
            assert (scales > 0.0).all(), "Error!"

        scales = scales + self.scales_min

        # [log(scales_min), log(scales_max + scales_min)]
        idx = torch.log(scales)
        # [0, log(scales_max + scales_min) - log(scales_min)]
        idx = idx - math.log(self.scales_min)
        normalizer = self.scales_max / self.scales_min  # [0, 1]
        normalizer = normalizer + 1  # log(max + min) - log(min) = log(max/min + 1)
        idx = idx / math.log(normalizer)
        # [0, num_bins-1]
        idx = idx * (self.num_bins - 1)
        if not training:
            idx = torch.round(idx)
        return idx

    def from_scale_idx(self, idx: Tensor) -> Tensor:
        """Convert index to scale"""
        out = idx / (self.num_bins - 1)
        normalizer = self.scales_max / self.scales_min
        normalizer = normalizer + 1  # log(max+min) - log(min) = log(max / min + 1)
        out = out * math.log(normalizer)
        return torch.exp(out + math.log(self.scales_min))


class ContinuousIndexedGaussianConditional(EntropyModel):
    """
    Gaussian Conditional prior with mean and scale indexes
    """

    def __init__(
        self,
        index_ranges: Tuple[int, int],  # [n_means, n_scales]
        *args: Any,
        scale_bound: float = 0.01,
        tail_mass: float = (2 ** (-8)),
        precision: int = 12,
        scaler: Scaler = Scaler(num_bins=256, scales_min=0.01),
        **kwargs: Any,
    ):
        """
        Args:
            index_ranges: a tuple (num_means, num_scales)
            scale_bound: scale lower bound, must be positive. Defaults to 0.01.
            tail_mass: tail mass. Defaults to (2 ** (-8)).
            precision: precision for the entropy coder. Defaults to 12.
            scaler: helper to map scales to indexes, see `Scaler`.
                Defaults to Scaler(num_bins=256, scales_min=0.01).
        """
        super().__init__(*args, **kwargs)
        self.precision = precision
        self._index_ranges = tuple(int(r) for r in index_ranges)
        self._num_means = self._index_ranges[0]
        self._not_one_mean = self._num_means != 1
        self._num_scales = self._index_ranges[1]

        if scaler.num_bins != self._num_scales:
            raise ValueError("Number of bins in `scaler` must match number of scales")
        self._scaler = scaler
        self.tail_mass = float(tail_mass)

        if scale_bound <= 0:
            raise ValueError("Invalid parameters")
        self.lower_bound_scale = LowerBound(scale_bound)
        self.lower_bound_zero = LowerBound(0.0)
        # TODO make an actual upperbound:
        self.upper_bound_mean = LowerBound(-float(index_ranges[0] - 1))
        self.upper_bound_scale = LowerBound(-float(index_ranges[1] - 1))

        self.register_buffer("scale_bound", torch.Tensor([float(scale_bound)]))
        self.register_buffer("_indexes_table", torch.Tensor())  # [n_means, n_scales, 2]
        # the tensors _prior_mean  _prior_scale and will be of shape [n_means, n_scales]
        # after runing `update tables`. All means (scales) are equal across columns (rows)
        # and are in an increasing order.
        self.register_buffer("_prior_mean", torch.Tensor())
        self.register_buffer("_prior_scale", torch.Tensor())
        self._prior_dtype = torch.float32

    def _get_prior_params(
        self, means_idx: Tensor, scales_idx: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Map indexes to the parameters of the gaussian prior

        Args:
            means_idx: tensor of mean indexes
            scales_idx: tensor of scale indexes

        Returns:
            Tuple (means, scales)
        """
        scales = self._scaler.from_scale_idx(scales_idx)
        means = means_idx / self._num_means - 0.5 * self._not_one_mean
        return means, scales

    @staticmethod
    def _prepare_table(table: Tensor):
        # convert to tensor of floats
        return table.to(torch.float32)

    def _standardized_cumulative(self, inputs: Tensor) -> Tensor:
        half = float(0.5)
        const = float(-(2**-0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)

    @staticmethod
    def _standardized_quantile(quantile):
        return sps.norm.ppf(quantile)

    def update_tables(self, force: bool = True) -> bool:
        # Check if we need to update the gaussian conditional parameters, the
        # offsets are only computed and stored when the conditonal model is updated
        if self._offset.numel() > 0 and not force:
            return False
        device = self._indexes_table.device

        indexes = [
            torch.arange(0, r, dtype=torch.int32, device=device)
            for r in self._index_ranges
        ]  # [means, scales]
        indexes = torch.meshgrid(*indexes, indexing="ij")
        indexes = torch.stack(indexes, dim=-1)  # [n_means, n_scales, 2]
        self._indexes_table = self._prepare_table(table=indexes).to(device)

        # Compute the mean and scale grids of the quantized prior
        _prior_mean, _prior_scale = self._get_prior_params(
            indexes[..., 0], indexes[..., 1]
        )
        # sanity check: grids are increaing
        assert _prior_mean[:, 0].equal(torch.sort(_prior_mean[:, 0]).values)
        assert _prior_scale[0].equal(torch.sort(_prior_scale[0]).values)

        self._prior_mean = _prior_mean
        self._prior_scale = _prior_scale
        self._build_tables()

        return True

    def _build_tables(self) -> None:
        """
        Build the quantized CDF table, which will contain num_means*num_scales PMFs
        """
        device = self._indexes_table.device
        sps_norm = sps.norm(self._prior_mean.numpy(), self._prior_scale.numpy())
        tail_lower = torch.tensor(sps_norm.ppf(self.tail_mass / 2), device=device)
        tail_upper = torch.tensor(sps_norm.ppf(1.0 - self.tail_mass / 2), device=device)

        minimas = torch.floor(tail_lower).to(torch.int32)  # [num_means, num_scales]
        maximas = torch.ceil(tail_upper).to(torch.int32)  # [num_means, num_scales]
        pmf_start = minimas.to(self._prior_dtype)  # [num_means, num_scales]
        pmf_length = maximas - minimas + 1  # [num_means, num_scales]
        max_length = int(torch.max(pmf_length).item())

        samples = torch.arange(0, max_length).to(self._prior_dtype)  # [max_len]
        samples = samples.unsqueeze(-1).unsqueeze(-1).to(device)  # [max_len, 1, 1]
        samples = samples + pmf_start.unsqueeze(0)  # [max_len, num_means, num_scales]

        num_means, num_scales = self._index_ranges
        num_pmfs = num_means * num_scales
        pmf_length = pmf_length.reshape(num_pmfs)  # sum(pmf_length) -> relates to CDFs
        cdf_offset = minimas.reshape(num_pmfs)

        upper = self._standardized_cumulative(
            (0.5 - (samples - self._prior_mean.numpy()).abs())
            / self._prior_scale.numpy()
        )  # [max_len, num_means, num_scales]
        lower = self._standardized_cumulative(
            (-0.5 - (samples - self._prior_mean.numpy()).abs())
            / self._prior_scale.numpy()
        )  # [max_len, num_means, num_scales]
        pmf = upper - lower  #
        pmf = pmf.reshape(max_length, num_pmfs).T  # [num_pmfs, max_length]

        tail_mass = 2 * lower.reshape(max_length, num_pmfs).T[:, :1]

        quantized_cdf = torch.Tensor(num_pmfs, max_length + 2, device=device)
        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        # (quantized_cdf!=0).float().sum() == (sum(pmf_length) + len(tail_mass))

        self._quantized_cdf = quantized_cdf
        self._offset = cdf_offset  # -pmf_center
        self._cdf_length = pmf_length + 2

    def _likelihood(self, inputs: Tensor, indexes: Tensor) -> Tensor:
        """
        Compute the likelihood of the convolved (Normal + Uniform noise)
        """
        half = float(0.5)
        means, scales = self._get_prior_params(
            means_idx=indexes[..., 0], scales_idx=indexes[..., 1]
        )
        values = torch.abs(inputs - means)

        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower

        return likelihood

    def _normalize_indexes(self, indexes: Tensor) -> Tensor:
        """
        Ensure indexes are within the allowed bounds in a differentiable way
        """
        indexes = self.lower_bound_zero(indexes)
        # NOTE: these are actually LowerBounds, so use -f(-input), see `__init__`
        i0 = -self.upper_bound_mean(-indexes[..., 0])
        i1 = -self.upper_bound_scale(-indexes[..., 1])
        return torch.stack([i0, i1], dim=-1)

    def forward(
        self,
        inputs: Tensor,
        indexes: Tensor,
        training: Optional[bool] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass: add uniform noise to inputs, compute their corresponding likelihood

        Args:
            inputs: tensor to be noised/quantized
            indexes: indexes of means and scales
            training: whether we are training. Defaults to None.

        Returns:
            tuple of tensors: noised input, its likelihood
        """
        training = self.training if training is None else training
        indexes = self._normalize_indexes(indexes)
        outputs = self.quantize(inputs, "noise" if training else "dequantize")
        likelihood = self._likelihood(outputs, indexes)

        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        return outputs, likelihood

    @staticmethod
    def _find_idx_of_first_smallest(inputs: Tensor, grid: Tensor) -> torch.IntTensor:
        return torch.max(
            (inputs.unsqueeze(-1) <= grid.view([1] * inputs.dim() + [len(grid)])),
            dim=-1,
        ).indices

    def indexes_to_cdf_indexes(
        self, indexes_mean: Tensor, indexes_scale: Tensor
    ) -> Tensor:
        """
        Convert indexes to CDF table indexes

        Args:
            indexes_mean: mean indexes (ints or floats of ints)
            indexes_scale: scale indexes (ints or floats of ints)

        Returns:
            tensor of ints containing the actual CDF table indexes
        """
        # cdf is pmf_length = num_means*num_scales long, where means and scalea are
        # arranged as [mean_0]*num_means, all scales, [mean_1]*num_means, all scales ...
        # so the correct indexing is:
        cdf_indexes = indexes_mean.int() * self._num_scales + indexes_scale.int()
        return cdf_indexes  # Ints

    def build_indexes(self, params: Tuple[Tensor, Tensor]) -> Tensor:
        """
        Convert prior parameters to CDF table indexes

        Args:
            params: tuple of tensors (means, scales)

        Returns:
            tensor of ints containing the actual CDF table indexes
        """
        means, scales = params
        mean_grid = self._prior_mean[:, 0]  # all means equal along cols
        scale_grid = self._prior_scale[0]  # all scales equal along rows

        indexes_mean = self._find_idx_of_first_smallest(inputs=means, grid=mean_grid)
        indexes_scale = self._find_idx_of_first_smallest(inputs=scales, grid=scale_grid)
        return self.indexes_to_cdf_indexes(indexes_mean, indexes_scale)


class GsnConditionalLocScaleShift(nn.Module):
    """
    Coding `input - round(mean)` using `mean - round(mean)`
    """

    def __init__(
        self,
        num_scales=256,
        min_scale=0.01,
        num_means=100,
        tail_mass: float = (2 ** (-8)),
        round_idx: bool = True,
    ) -> None:
        super().__init__()
        self._scaler = Scaler(num_bins=num_scales, scales_min=min_scale)
        self._num_means = num_means
        self._round_idx = round_idx
        self._num_scales = num_scales
        self._min_scale = min_scale
        self._one_mean_flag = num_means == 1

        self._entropy_model = ContinuousIndexedGaussianConditional(
            index_ranges=(num_means, num_scales),
            tail_mass=tail_mass,
            scale_bound=min_scale,
            scaler=self._scaler,
            entropy_coder_precision=16,  # default
        )

    @staticmethod
    def _round_st(x: Tensor) -> Tensor:
        """Straight-through round"""
        return (torch.round(x) - x).detach() + x

    @staticmethod
    def verysoftplus(inputs):
        """Evaluates a softplus function transitioning from 1/(1-x) to 1+x at x=0."""
        inputs_pos = torch.maximum(inputs, torch.zeros_like(inputs))
        inputs_neg = torch.minimum(inputs, torch.zeros_like(inputs))
        return torch.where(inputs > 0, inputs_pos + 1.0, 1.0 / (1.0 - inputs_neg))

    def _get_indexes(
        self, means: Tensor, scales: Tensor, training: bool = True
    ) -> Tensor:
        """
        Get indexes from means and scales
        """
        scales_i = self._scaler.to_scale_idx(
            self.verysoftplus(scales), training=training
        )
        mean_i = (means - self._round_st(means) + 0.5) * self._num_means
        return torch.stack([mean_i, scales_i], dim=-1)

    def forward(
        self,
        inputs: Tensor,
        scales: Tensor,
        means: Tensor,
        training: Optional[bool] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compress inputs, return its quantized version and number of bits

        Args:
            inputs: input tensor (floats)
            scales: tensor (floats) of scales
            means: tensor (floats) of means
            training: whether we are in train mode. Defaults to None.

        Returns:
            Tuple of tensors: quantized inputs and bits
        """
        training = self.training if training is None else training

        # flaot means and scales to indexes where mean is (mean - round(mean))
        indexes = self._get_indexes(means, scales, training=training)

        round_mean = self._round_st(means)
        inputs = inputs - round_mean

        quantized_inputs, bits = self._entropy_model(
            inputs, indexes=indexes, training=training
        )  # noised input to get approximate rate
        if training:
            quantized_inputs = self._round_st(inputs)  # Round + STE

        return quantized_inputs + round_mean, bits

    def compress(
        self, inputs: Tensor, scales: Tensor, means: Tensor
    ) -> Tuple[Tensor, List[str]]:
        """
        Compress `inputs` tensor to string

        Args:
            inputs: input tensor (floats) to compress
            scales: tensor (floats) of scales
            means: tensor (floats) of means

        Returns:
            Quantized input and compressed string
        """
        indexes = self._get_indexes(means, scales)
        indexes = self._entropy_model._normalize_indexes(indexes)
        cdf_indexes = self._entropy_model.indexes_to_cdf_indexes(
            indexes_mean=indexes[..., 0], indexes_scale=indexes[..., 1]
        )  # Ints

        round_mean = torch.round(means)
        inputs = inputs - round_mean  # we are compressing this object

        bytestring = self._entropy_model.compress(inputs=inputs, indexes=cdf_indexes)
        quantized = torch.round(inputs)
        return quantized + round_mean, bytestring

    def decompress(self, strings: str, scales: Tensor, means: Tensor) -> Tensor:
        """
        Decompress character strings to Tensors.
        """
        indexes = self._get_indexes(means, scales)
        indexes = self._entropy_model._normalize_indexes(indexes)
        cdf_indexes = self._entropy_model.indexes_to_cdf_indexes(
            indexes_mean=indexes[..., 0], indexes_scale=indexes[..., 1]
        )  # Ints
        round_mean = torch.round(means)
        outputs = self._entropy_model.decompress(
            strings,
            cdf_indexes,  # type: ignore # expects IntTensor, whcih cdf_indexes is
        )
        return outputs + round_mean

    def update(self, force: bool = True) -> bool:
        bottleneck_updated = self._entropy_model.update_tables(force=force)
        return bottleneck_updated
