"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import abc
from abc import ABCMeta
from typing import Optional, Tuple

import torch
from torch import IntTensor, Size, Tensor
from torch.distributions import Distribution, Laplace
from torch.nn import Module


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
        cdf: If provided, used for range coding rather than the probability
            tables built from the prior.
        cdf_offset: Must be provided alongside ``cdf``.
        cdf_size: Must be provided alongside ``cdf``.
        maximum_cdf_size: The maximum ``cdf_size``. When provided, an empty
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
        cdf: Optional[IntTensor] = None,
        cdf_offset: Optional[IntTensor] = None,
        cdf_size: Optional[IntTensor] = None,
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
            if not (cdf is None) == (cdf_offset is None) == (cdf_size is None):
                error_message = """
                either all or none of the cumulative distribution function 
                (CDF) arguments (`cdf`, `cdf_offset`, `cdf_size`, and 
                `maximum_cdf_size`) must be provided                
                """

                raise ValueError(error_message)

            if (prior is None) + (maximum_cdf_size is None) + (cdf is None) != 2:
                error_message = """
                when `compression` is `True`, exactly one of `prior`, `cdf`, or 
                `maximum_cdf_size` must be provided. 
                """

                raise ValueError(error_message)

            if prior is not None:
                # TODO(@0x00b1): pre-compute probability tables
                pass
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

                cdf = zeros

                cdf_offset = zeros[:, 0]
                cdf_size = zeros[:, 0]

            self._cdf = cdf

            self._cdf_offset = cdf_offset

            self._cdf_size = cdf_size

            if not self.stateless:
                if self._cdf is not None:
                    self._cdf = self.cdf.detach()

                if self._cdf_offset is not None:
                    self._cdf_offset = self.cdf_offset.detach()

                if self._cdf_size is not None:
                    self._cdf_size = self.cdf_size.detach()
        else:
            if not (
                cdf is None
                and cdf_offset is None
                and cdf_size is None
                and maximum_cdf_size is None
            ):
                error_message = """
                cumulative distribution function (CDF) arguments (`cdf`, 
                `cdf_offset`, `cdf_size`, and `maximum_cdf_size`) cannot be 
                provided when `compression` is `False`
                """

                raise ValueError(error_message)

        if laplace_tail_mass:
            self._laplace_prior = Laplace(0.0, 1.0)

        self._laplace_tail_mass = laplace_tail_mass

    @property
    def cdf(self):
        self._check_compression()

        return torch.tensor(self._cdf)

    @property
    def cdf_offset(self):
        self._check_compression()

        return torch.tensor(self._cdf_offset)

    @property
    def cdf_size(self):
        self._check_compression()

        return torch.tensor(self._cdf_size)

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
