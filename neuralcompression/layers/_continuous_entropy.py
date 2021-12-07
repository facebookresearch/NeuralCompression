# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import Optional, Tuple, Union

import torch
import torch.nn
from torch import IntTensor, Size, Tensor
from torch.distributions import Distribution
from torch.nn import Module

import neuralcompression.functional as ncF
from neuralcompression.distributions import UniformNoise


class ContinuousEntropy(Module):
    r"""Abstract base class (ABC) for continuous entropy layers

    This base class pre-computes integer probability tables based on a prior
    distribution, which can be used across different platforms by a range
    encoder and decoder.

    Args:
        coding_rank: Number of innermost dimensions considered a coding unit.
            Each coding unit is compressed to its own bit string. The coding
            units are summed during propagation.
        compressible: If ``True``, the integer probability tables used by the
            ``compress()`` and ``decompress()`` methods will be instantiated.
            If ``False``, these two methods are inaccessible.
        stateless: If ``True``, creates range coding tables as ``Tensor``s.
            This makes the entropy model stateless and the range coding tables
            are expected to be provided manually. If ``compressible`` is
            ``False``, then it is implied ``stateless`` is ``True`` and the
            provided ``stateless`` value is ignored. If ``False``, range coding
            tables are persisted as ``Parameter``s. This allows the entropy
            model to be serialized, so both the range encoder and decoder use
            identical tables when loading the stored model.
        prior: A probability density function fitting the marginal
            distribution of the bottleneck data with additive uniform noise,
            which is shared a priori between the sender and the receiver. For
            best results, the distribution should be flexible enough to have a
            unit-width uniform distribution as a special case, since this is
            the marginal distribution for bottleneck dimensions that are
            constant.
        tail_mass: An approximate probability mass which is range encoded with
            less precision, by using a Golomb-like code.
        prior_dtype: The data type of the prior. Must be provided when
            ``prior`` is omitted.
        prior_shape: The batch shape of the prior, i.e. dimensions which are
            not assumed identically distributed (i.i.d.). Must be provided when
            ``prior`` is omitted.
        range_coder_precision: The precision passed to ``range_encoder`` and
            ``range_decoder``.
        cdfs: If provided, used for range coding rather than the probability
            tables built from ``prior``.
        cdf_sizes: Must be provided alongside ``cdfs``.
        cdf_offsets: Must be provided alongside ``cdfs``.
        maximum_cdf_size: The maximum ``cdf_sizes``. When provided, an empty
            range coding table is created, which can then be restored. Requires
            ``compressible`` to be ``True`` and ``stateless`` to be ``False``.
    """

    _coding_rank: Optional[int]

    _cdfs: IntTensor
    _cdf_sizes: IntTensor
    _cdf_offsets: IntTensor

    def __init__(
        self,
        coding_rank: Optional[int] = None,
        compressible: bool = False,
        stateless: bool = False,
        prior: Optional[Union[Distribution, UniformNoise]] = None,
        tail_mass: float = 2 ** -8,
        prior_shape: Optional[Tuple[int, ...]] = None,
        prior_dtype: Optional[torch.dtype] = None,
        cdfs: Optional[IntTensor] = None,
        cdf_sizes: Optional[IntTensor] = None,
        cdf_offsets: Optional[IntTensor] = None,
        maximum_cdf_size: Optional[int] = None,
        range_coder_precision: int = 12,
    ):
        super(ContinuousEntropy, self).__init__()

        self._coding_rank = coding_rank

        self._compressible = compressible

        self._stateless = stateless

        self._tail_mass = tail_mass

        self._range_coder_precision = range_coder_precision

        if prior is None:
            if prior_shape is None or prior_dtype is None:
                error_message = r"""either `prior` or both `prior_dtype` and
                `prior_shape` must be provided
                """

                raise ValueError(error_message)

            self._prior_shape = Size(prior_shape)
            self._prior_dtype = prior_dtype
        else:
            self._prior_shape = prior.batch_shape
            self._prior_dtype = torch.float32

        if self.compressible:
            if not (cdfs is None) == (cdf_sizes is None) == (cdf_offsets is None):
                error_message = r"""either all or none of the cumulative
                distribution function (CDF) arguments (`cdfs`, `cdf_offsets`,
                `cdf_sizes`, and `maximum_cdf_size`) must be provided
                """

                raise ValueError(error_message)

            if (prior is None) + (maximum_cdf_size is None) + (cdfs is None) != 2:
                error_message = r"""when `compressible` is `True`, exactly one
                of `prior`, `cdfs`, or `maximum_cdf_size` must be provided.
                """

                raise ValueError(error_message)

            if prior is not None:
                self._prior = prior

                cdfs, cdf_sizes, cdf_offsets = self._update()
            elif maximum_cdf_size is not None:
                if self.stateless:
                    error_message = r"""if `stateless` is `True`, cannot
                    provide `maximum_cdf_size`
                    """

                    raise ValueError(error_message)

                context_size = len(self.context_shape)

                zeros = torch.zeros(
                    [context_size, maximum_cdf_size],
                    dtype=torch.int32,
                )

                cdfs = IntTensor(zeros)

                cdf_offsets = IntTensor(zeros[0, :])
                cdf_sizes = IntTensor(zeros[0, :])
        else:
            if not (
                cdfs is None
                and cdf_offsets is None
                and cdf_sizes is None
                and maximum_cdf_size is None
            ):
                error_message = r"""cumulative distribution function (CDF)
                arguments (`cdfs`, `cdf_offsets`, `cdf_sizes`, and
                `maximum_cdf_size`) cannot be provided when `compressible` is
                `False`
                """

                raise ValueError(error_message)

        self.register_buffer("_cdfs", cdfs)
        self.register_buffer("_cdf_sizes", cdf_sizes)
        self.register_buffer("_cdf_offsets", cdf_offsets)

        self._maximum_cdf_size = maximum_cdf_size

    @property
    def cdf_offsets(self) -> IntTensor:
        self._validate_compress()

        return self._cdf_offsets

    @property
    def cdf_sizes(self) -> IntTensor:
        self._validate_compress()

        return self._cdf_sizes

    @property
    def cdfs(self) -> IntTensor:
        self._validate_compress()

        return self._cdfs

    @property
    def coding_rank(self) -> Optional[int]:
        """Number of innermost dimensions considered a coding unit."""
        return self._coding_rank

    @property
    def compressible(self) -> bool:
        """If ``True``, the integer probability tables used by the
        ``compress()`` and ``decompress()`` methods have been instantiated and
        the layer is prepared for compression.
        """
        return self._compressible

    @property
    def context_shape(self) -> Size:
        """The shape of the non-flattened probability density function (PDF)
        and cumulative distribution function (CDF) range coding tables.

        Typically equal to ``prior_shape``, but may and can differ. Regardless,
        ``context_shape`` contains the ``prior_shape`` in its trailing
        dimensions.
        """
        return self._prior_shape

    @property
    def prior(self) -> Union[Distribution, UniformNoise]:
        """A prior distribution, used for computing range coding tables."""
        if self._prior is None:
            error_message = r"""instance doesn’t hold a reference to its prior
            distribution
            """

            raise RuntimeError(error_message)

        return self._prior

    @prior.deleter
    def prior(self):
        self._prior = None

    @property
    def prior_dtype(self) -> torch.dtype:
        """The data type of ``prior``."""
        return self._prior_dtype

    @property
    def prior_shape(self) -> Size:
        """Batch shape of ``prior``, dimensions which are **not** assumed
        independent and identically distributed (i.i.d.).
        """
        return self._prior_shape

    @property
    def range_coder_precision(self) -> int:
        """The precision passed to ``range_encoder`` and ``range_decoder``."""
        return self._range_coder_precision

    @property
    def stateless(self) -> bool:
        return self._stateless

    @property
    def tail_mass(self) -> float:
        """An approximate probability mass which is range encoded with less
        precision, by using a Golomb-like code.
        """
        return self._tail_mass

    @abc.abstractmethod
    def compress(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def decompress(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def quantize(
        bottleneck: Tensor,
        indexes: Tensor,
        offsets: Optional[Tensor] = None,
    ) -> IntTensor:
        """Quantizes a floating-point ``Tensor``.

        To use this entropy layer as an information bottleneck during training,
        pass a ``Tensor`` to this function. The ``Tensor`` is rounded to
        integer values modulo a quantization offset, which depends on
        ``indexes``. For example, for a ``Normal`` distribution, the returned
        values are rounded to the location of the mode of the distributions
        plus or minus an integer.

        The gradient of this rounding operation is overridden with the identity
        (straight-through gradient estimator).

        Args:
            bottleneck: the data to be quantized.
            indexes: the scalar distribution for each element in
                ``bottleneck``.
            offsets:

        Returns:
            the quantized values.
        """
        outputs = bottleneck.clone()

        if offsets is not None:
            outputs = outputs - offsets

        outputs = torch.round(outputs)

        return IntTensor(outputs)

    def _update(self):
        quantization_offset = ncF.quantization_offset(self.prior)

        if isinstance(self.prior, UniformNoise):
            lower_tail = self.prior.lower_tail(self.tail_mass)
        else:
            lower_tail = ncF.lower_tail(self.prior, self.tail_mass)

        if isinstance(self.prior, UniformNoise):
            upper_tail = self.prior.upper_tail(self.tail_mass)
        else:
            upper_tail = ncF.upper_tail(self.prior, self.tail_mass)

        minimum = torch.floor(lower_tail - quantization_offset)
        minimum = minimum.to(torch.int32)
        minimum = torch.clamp_min(minimum, 0)

        maximum = torch.ceil(upper_tail - quantization_offset)
        maximum = maximum.to(torch.int32)
        maximum = torch.clamp_min(maximum, 0)

        pmf_start = minimum.to(self.prior_dtype) + quantization_offset
        pmf_start = pmf_start.to(torch.int32)

        pmf_sizes = maximum - minimum + 1

        maximum_pmf_size = torch.max(pmf_sizes).to(self.prior_dtype)
        maximum_pmf_size = maximum_pmf_size.to(torch.int32)

        samples = torch.arange(maximum_pmf_size).to(self.prior_dtype)
        samples = samples.reshape([-1] + len(self.context_shape) * [1])
        samples = samples + pmf_start

        if isinstance(self.prior, UniformNoise):
            pmfs = self.prior.prob(samples)
        else:
            pmfs = torch.exp(self.prior.log_prob(samples))

        pmf_sizes = torch.broadcast_to(pmf_sizes, self.context_shape)
        pmf_sizes = pmf_sizes.squeeze()

        cdf_sizes = pmf_sizes + 2

        cdf_offsets = torch.broadcast_to(minimum, self.context_shape)
        cdf_offsets = cdf_offsets.squeeze()

        cdfs = torch.zeros(
            (len(pmf_sizes), int(maximum_pmf_size) + 2),
            dtype=torch.int32,
            device=pmfs.device,
        )

        for index, (pmf, pmf_size) in enumerate(zip(pmfs, pmf_sizes)):
            pmf = pmf[:pmf_size]

            overflow = torch.clamp(
                1 - torch.sum(pmf, dim=0, keepdim=True),
                min=0.0,
            )

            pmf = torch.cat([pmf, overflow], dim=0)

            cdf = ncF.pmf_to_quantized_cdf(
                pmf,
                self._range_coder_precision,
            )

            cdfs[index, : cdf.size()[0]] = cdf

        return cdfs, cdf_sizes, cdf_offsets

    def _validate_cdf_offsets(self):
        offsets = self._cdf_offsets

        if offsets.numel() == 0:
            error_message = r"""uninitialized cumulative distribution function
            (CDF) offsets
            """

            raise ValueError(error_message)

        if len(offsets.size()) != 1:
            error_message = r"""invalid cumulative distribution function (CDF)
            offsets
            """

            raise ValueError(error_message)

    def _validate_cdf_sizes(self):
        sizes = self._cdf_sizes

        if sizes.numel() == 0:
            error_message = r"""uninitialized cumulative distribution function
            (CDF) sizes
            """

            raise ValueError(error_message)

        if len(sizes.size()) != 1:
            error_message = r"""invalid cumulative distribution function (CDF)
            sizes
            """

            raise ValueError(error_message)

    def _validate_cdfs(self):
        functions = self._cdfs

        if functions.numel() == 0:
            error_message = r"""uninitialized cumulative distribution functions
            (CDFs)
            """

            raise ValueError(error_message)

        if len(functions.size()) != 2:
            error_message = r"""invalid ``size()`` of cumulative distribution
            functions (CDFs)
            """

            raise ValueError(error_message)

    def _validate_compress(self):
        if not self.compressible:
            error_message = r"""for range coding, `compress` must be `True`
            """

            raise RuntimeError(error_message)

    def _validate_range_coding_table(self):
        self._validate_cdf_offsets()
        self._validate_cdf_sizes()
        self._validate_cdfs()
