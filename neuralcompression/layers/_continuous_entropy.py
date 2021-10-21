"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import abc
from abc import ABCMeta
from typing import Optional, Tuple

import torch
from compressai.entropy_models.entropy_models import pmf_to_quantized_cdf
from torch import IntTensor, Size, Tensor
from torch.distributions import Distribution, Laplace
from torch.nn import Module, functional as F

import neuralcompression.functional as ncF


class ContinuousEntropy(Module, metaclass=ABCMeta):
    """Abstract base class (ABC) for implementing continuous entropy layers.

    The class pre-computes integer probability tables based on a prior
    distribution, which can be used across different platforms by a range
    encoder and decoder.

    Args:
        prior: A probability density function fitting the marginal
            distribution of the bottleneck data with additive uniform noise,
            which is shared a priori between the sender and the receiver. For
            best results, the distribution should be flexible enough to have a
            unit-width uniform distribution as a special case, since this is
            the marginal distribution for bottleneck dimensions that are
            constant.
        coding_rank: Number of innermost dimensions considered a coding unit.
            Each coding unit is compressed to its own bit string. The coding
            units are summed during propagation.
        compression: If ``True``, the integer probability tables used by the
            ``compress()`` and ``decompress()`` methods will be instantiated.
            If ``False``, these two methods will be inaccessible.
        stateless: If `True`, creates range coding tables as ``Tensor``s. This
            makes the entropy model stateless and the range coding tables are
            expected to be provided manually. If ``compression`` is ``False``,
            then it is implied ``stateless`` is ``True`` and the provided
            ``stateless`` value is ignored. If ``False``, range coding tables
            are persisted as ``Parameter``s. This allows the entropy model to
            be serialized, so both the range encoder and decoder use identical
            tables when loading the stored model.
        tail_mass: An approximate probability mass which is range encoded with
            less precision, by using a Golomb-like code.
        range_coding_precision: The precision passed to range encoder and
            decoder operations.
        prior_dtype: The data type of the prior. Must be provided when
            ``prior`` is omitted.
        prior_shape: The batch shape of the prior, i.e. dimensions which are
            not assumed identically distributed (i.i.d.). Must be provided when
            ``prior`` is omitted.
        cdfs: If provided, used for range coding rather than the probability
            tables built from the prior.
        cdf_offsets: Must be provided alongside ``cdfs``.
        cdf_sizes: Must be provided alongside ``cdfs``.
        maximum_cdf_size: The maximum ``cdf_sizes``. When provided, an empty
            range coding table is created, which can then be restored. Requires
            ``comprsssion`` to be ``True`` and ``stateless`` to be ``False``.
        laplace_tail_mass: If positive, augments the prior with a Laplace
            mixture for training stability.
    """

    def __init__(
        self,
        prior: Optional[Distribution] = None,
        coding_rank: Optional[int] = None,
        compression: bool = False,
        stateless: bool = False,
        tail_mass: float = 2 ** -8,
        range_coding_precision: int = 12,
        prior_dtype: Optional[torch.dtype] = None,
        prior_shape: Optional[Tuple[int, ...]] = None,
        cdfs: Optional[IntTensor] = None,
        cdf_offsets: Optional[IntTensor] = None,
        cdf_sizes: Optional[IntTensor] = None,
        maximum_cdf_size: Optional[int] = None,
        laplace_tail_mass: float = 0.0,
    ):
        super(ContinuousEntropy, self).__init__()

        self._prior = prior

        self._coding_rank = coding_rank

        self._compression = compression

        self._stateless = stateless

        self._tail_mass = tail_mass

        self._range_coder_precision = range_coding_precision

        if prior is None:
            if prior_dtype is None and prior_shape is None:
                error_message = """
                either `prior` or both `dtype` and `prior_shape` must be 
                provided
                """

                raise ValueError(error_message)

            self._prior_dtype = prior_dtype

            self._prior_shape = Size(prior_shape)
        else:
            self._prior_dtype = torch.dtype

            self._prior_shape = prior.batch_shape

        if self.compression:
            if not (cdfs is None) == (cdf_offsets is None) == (cdf_sizes is None):
                error_message = """
                either all or none of the cumulative distribution function 
                (CDF) arguments (`cdfs`, `cdf_offsets`, `cdf_sizes`, and 
                `maximum_cdf_size`) must be provided                
                """

                raise ValueError(error_message)

            if (prior is None) + (maximum_cdf_size is None) + (cdfs is None) != 2:
                error_message = """
                when `compression` is `True`, exactly one of `prior`, `cdfs`, 
                or `maximum_cdf_size` must be provided. 
                """

                raise ValueError(error_message)

            if prior is not None:
                cdfs, cdf_sizes, cdf_offsets = self._precompute_probability_table()
            elif maximum_cdf_size is not None:
                if self.stateless:
                    error_message = """
                    if `stateless` is `True`, cannot provide `maximum_cdf_size`
                    """

                    raise ValueError(error_message)

                context_size = self.context_shape.num_elements()

                zeros = torch.zeros(
                    [context_size, maximum_cdf_size],
                    dtype=torch.int32,
                )

                cdfs = zeros

                cdf_offsets = zeros[:, 0]
                cdf_sizes = zeros[:, 0]

            self._cdfs = cdfs

            self._cdf_offsets = cdf_offsets

            self._cdf_sizes = cdf_sizes

            if not self.stateless:
                if self.cdfs is not None:
                    self._cdfs = self.cdfs.detach()

                if self.cdf_offsets is not None:
                    self._cdf_offsets = self.cdf_offsets.detach()

                if self.cdf_offsets is not None:
                    self._cdf_sizes = self.cdf_sizes.detach()
        else:
            if not (
                cdfs is None
                and cdf_offsets is None
                and cdf_sizes is None
                and maximum_cdf_size is None
            ):
                error_message = """
                cumulative distribution function (CDF) arguments (`cdfs`, 
                `cdf_offsets`, `cdf_sizes`, and `maximum_cdf_size`) cannot be 
                provided when `compression` is `False`
                """

                raise ValueError(error_message)

        if laplace_tail_mass:
            self._laplace_prior = Laplace(0.0, 1.0)

        self._laplace_tail_mass = laplace_tail_mass

    @property
    def cdfs(self):
        self._check_compression()

        return torch.tensor(self._cdfs)

    @property
    def cdf_offsets(self):
        self._check_compression()

        return torch.tensor(self._cdf_offsets)

    @property
    def cdf_sizes(self):
        self._check_compression()

        return torch.tensor(self._cdf_sizes)

    @property
    def coding_rank(self):
        return self._coding_rank

    @property
    def compression(self):
        return self._compression

    @property
    def context_shape(self):
        return self.prior_shape

    @property
    def context_shape_tensor(self):
        return torch.tensor(self.context_shape, dtype=torch.int32)

    @property
    def laplace_tail_mass(self) -> float:
        return self._laplace_tail_mass

    @property
    def prior(self):
        if self._prior is None:
            error_message = """
            This entropy model doesn't hold a reference to its prior 
            distribution. This can happen depending on how it is instantiated, 
            e.g., if it is unserialized.
            """

            raise RuntimeError(error_message)

        return self._prior

    @prior.deleter
    def prior(self):
        self._prior = None

    @property
    def prior_dtype(self):
        return self._prior_dtype

    @property
    def prior_shape(self):
        return self._prior_shape

    @property
    def prior_shape_tensor(self):
        return torch.tensor(self.prior_shape, dtype=torch.int32)

    @property
    def range_coder_precision(self):
        return self._range_coder_precision

    @property
    def stateless(self):
        return self._stateless

    @property
    def tail_mass(self):
        return self._tail_mass

    @abc.abstractmethod
    def compress(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def decompress(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def quantize(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def reconstruct(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def _check_compression(self):
        if not self.compression:
            error_message = """
            for range coding, `compression` must be `True`
            """

            raise RuntimeError(error_message)

    def _precompute_probability_table(self):
        quantization_offset = ncF.quantization_offset(self.prior)

        lower_tail = ncF.lower_tail(
            self.prior,
            self.tail_mass,
        )

        upper_tail = ncF.upper_tail(
            self.prior,
            self.tail_mass,
        )

        minimum = torch.floor(
            lower_tail - quantization_offset,
        ).to(torch.int32)

        maximum = torch.ceil(
            upper_tail - quantization_offset,
        ).to(torch.int32)

        pmf_m = minimum.to(self.prior_dtype) + quantization_offset

        pmf_sizes = maximum - minimum + 1

        maximum_pmf_size = torch.max(pmf_sizes)

        samples = torch.arange(
            maximum_pmf_size.to(self.prior_dtype),
        ).to(self.prior_dtype)

        samples = torch.reshape(
            samples,
            [-1] + len(self.context_shape) * [1],
        )

        samples = samples + pmf_m

        pmfs = torch.reshape(
            torch.exp(self.prior.log_prob(samples)),
            [maximum_pmf_size, -1],
        )

        pmfs = torch.squeeze(pmfs)

        pmf_sizes = torch.broadcast_to(
            pmf_sizes,
            self.context_shape,
        )

        pmf_sizes = torch.reshape(pmf_sizes, [-1])

        cdf_sizes = pmf_sizes + 2

        cdf_offsets = torch.broadcast_to(
            minimum,
            self.context_shape,
        )

        cdf_offsets = torch.reshape(cdf_offsets, [-1])

        cdfs = torch.zeros(
            (len(pmf_sizes), maximum_pmf_size + 2),
            dtype=torch.int32,
        )

        for index, (pmf, pmf_size) in enumerate(zip(pmfs, pmf_sizes)):
            pmf = pmf[:pmf_size]

            overflow = torch.clamp_min(
                1.0 - torch.sum(pmf, 0, keepdim=True),
                0.0,
            )

            pmf = torch.cat((pmf, overflow), 0)

            quantized_cdf = pmf_to_quantized_cdf(
                pmf,
                self.range_coder_precision,
            )

            pad = (0, maximum_pmf_size - pmf_size)

            cdfs[index] = F.pad(quantized_cdf, pad, mode="constant", value=0)

        return cdfs, cdf_offsets, cdf_sizes
