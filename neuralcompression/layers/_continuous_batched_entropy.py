"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import functools
import typing
from typing import (
    List,
    Optional,
    Tuple,
    Union,
)

import torch
from compressai.ans import (
    RansDecoder,
    RansEncoder,
)
from torch import (
    IntTensor,
    Size,
    Tensor,
)
from torch.distributions import (
    Distribution,
)
from torch.nn import (
    Parameter,
)

import neuralcompression.functional as ncF
from neuralcompression.distributions import (
    UniformNoise,
)
from ._continuous_entropy import (
    ContinuousEntropy,
)


class ContinuousBatchedEntropy(ContinuousEntropy):
    """A batched entropy layer for continuous random variables.

    This entropy layer handles quantization of a ``bottleneck`` tensor and
    helps with training of the parameters of the probability distribution
    modeling the tensor (a shared "prior" between sender and receiver). It also
    pre-computes integer probability tables, which can then be used to compress
    and decompress bottleneck tensors reliably across different platforms.

    A typical workflow looks like this:

        1.  Train a model using an instance of this entropy layer as a
            bottleneck, passing the ``bottleneck`` tensor through it. With
            ``training`` is ``True``, the model computes a differentiable upper
            bound on the number of bits needed to compress the bottleneck
            tensor.

        2.  For evaluation, get a closer estimate of the number of compressed
            bits using `training=False`.

        3.  Instantiate an entropy model with `compression=True` (and the same
            parameters as during training), and share the model between a
            sender and a receiver.

        4.  On the sender side, compute the bottleneck tensor and call
            `compress()` on it. The output is a compressed string
            representation of the tensor. Transmit the string to the receiver,
            and call `decompress()` there. The output is the quantized
            bottleneck tensor. Continue processing the tensor on the receiving
            side.

    Entropy models which contain range coding tables (i.e. with
    `compression=True`) can be instantiated in two ways:

        1.  By providing a continuous "prior" distribution object. The range
            coding tables are then derived from that continuous distribution.

        3.  In a more low-level way, by directly providing the range coding
            tables to ``__init__``.

    This class assumes that all scalar elements of the encoded tensor are
    statistically independent, and that the parameters of their scalar
    distributions do not depend on data. The innermost dimensions of the
    bottleneck tensor must be broadcastable to the batch shape of `prior`. Any
    dimensions to the left of the batch shape are assumed to be i.i.d. (i.e.
    the likelihoods are broadcast to the bottleneck tensor accordingly).

    A more detailed description (and motivation) of this way of performing
    quantization and range coding can be found in the following paper. Please
    cite the paper when using this code for derivative work.

        | “End-to-end Optimized Image Compression”
        | Johannes Ballé, Valero Laparra, Eero P. Simoncelli
        | https://arxiv.org/abs/1611.01704

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

    _decoder = RansDecoder()
    _encoder = RansEncoder()

    def __init__(
        self,
        coding_rank: int,
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
        non_integer_offsets: bool = True,
        quantization_offset: Optional[IntTensor] = None,
    ):
        super(ContinuousBatchedEntropy, self).__init__(
            coding_rank=coding_rank,
            compressible=compressible,
            stateless=stateless,
            prior=prior,
            tail_mass=tail_mass,
            prior_shape=prior_shape,
            prior_dtype=prior_dtype,
            cdfs=cdfs,
            cdf_sizes=cdf_sizes,
            cdf_offsets=cdf_offsets,
            maximum_cdf_size=maximum_cdf_size,
            range_coder_precision=range_coder_precision,
        )

        self._non_integer_offsets = non_integer_offsets

        if self.coding_rank and self.coding_rank < len(self.prior_shape):
            error_message = "`coding_rank` can't be smaller than `prior_shape`"

            raise ValueError(error_message)

        _quantization_offset = None

        if not self.non_integer_offsets:
            _quantization_offset = None
        elif prior is not None:
            _quantization_offset = torch.broadcast_to(
                ncF.quantization_offset(self.prior),
                self.prior_shape,
            )
        elif maximum_cdf_size is not None:
            _quantization_offset = torch.zeros(
                self.prior_shape,
                dtype=self.prior_dtype,
            )
        else:
            assert cdfs is not None

            if quantization_offset is None:
                error_message = """must provide `quantization_offset` when
                `cdfs` is provided and `non_integer_offsets` is `True`"""

                raise ValueError(error_message)

        if _quantization_offset is None:
            self._quantization_offset = None
        elif self.compressible and not self.stateless:
            self.register_buffer(
                "_quantization_offset",
                torch.tensor(
                    _quantization_offset,
                    dtype=self.prior_dtype,
                ),
            )
        else:
            self.register_parameter(
                "_quantization_offset",
                Parameter(
                    torch.tensor(
                        _quantization_offset,
                        dtype=self.prior_dtype,
                    ),
                ),
            )

    @property
    def non_integer_offsets(self) -> bool:
        """Whether to quantize to non-integer offsets heuristically determined
        from the mode or median of ``self.prior``. ``False`` when using soft
        quantization during training.
        """
        return self.non_integer_offsets

    @property
    def quantization_offset(self) -> Optional[Tensor]:
        if self._quantization_offset is None:
            return None

        return torch.tensor(self._quantization_offset)

    def compress(self, bottleneck: Tensor) -> List[str]:
        """Compresses a floating-point tensor to bit strings.

        ``bottleneck`` is first quantized with the ``self.quantize`` method,
        then compressed using the probability tables (i.e. ``self.cdf_offsets``,
        ``self.cdf_sizes``, and ``self.cdfs`` properies) derived from
        ``self.prior``. The quantized tensor can be recovered by passing the
        compressed bit strings to the ``self.decompress`` method.

        The innermost ``self.coding_rank`` dimensions are treated as one coding
        unit (i.e. are compressed into one string each). Any additional
        dimensions to the left are treated as batch dimensions.

        Args:
            bottleneck: data to be compressed. Must have at least
            ``self.coding_rank`` dimensions, and the innermost dimensions must
            be broadcastable to ``self.prior_shape``.

        Returns:
            has the same shape as ``bottleneck`` without the
            ``self.coding_rank`` innermost dimensions, containing a string for
            each coding unit.
        """
        input_shape = torch.tensor(bottleneck.size(), dtype=torch.int32)

        batch_shape, coding_shape = torch.split(
            input_shape,
            [
                input_shape.size()[0] - self.coding_rank,
                self.coding_rank,
            ],
        )

        indexes = self._compute_indexes(
            coding_shape[: self.coding_rank - len(self.prior_shape)],
        )

        if self.quantization_offset is not None:
            bottleneck -= self.quantization_offset

        quantized = torch.reshape(
            torch.round(bottleneck).to(torch.int32),
            Size(
                torch.cat(
                    [torch.tensor([-1], dtype=torch.int32), coding_shape],
                    dim=0,
                ),
            ),
        )

        strings = []

        for index in range(quantized.size()[0]):
            string = self._encoder.encode_with_indexes(
                quantized[index].reshape(-1).to(torch.int32).tolist(),
                indexes[index].reshape(-1).to(torch.int32).tolist(),
                self.cdfs.tolist(),
                self.cdf_sizes.reshape(-1).to(torch.int32).tolist(),
                self.cdf_offsets.reshape(-1).to(torch.int32).tolist(),
            )

            strings += [string]

        return []

    def decompress(
        self,
        strings: Tensor,
        broadcast_shape: Tuple[int, ...],
    ):
        """Decompresses a tensor.

        Reconstructs the quantized tensor from bit strings emitted by the
        ``self.compress`` method. It is necessary to provide a part of the
        output shape as ``broadcast_shape``.

        Args:
            strings: the compressed bit strings.
            broadcast_shape: the part of the output tensor shape between the
            shape of ``strings`` on the left and ``self.prior_shape`` on the
            right. This must match the shape of the input passed to the
            ``self.compress`` method.

        Returns:
            has the shape
            ``(*strings.shape, *broadcast_shape, *self.prior_shape)``.
        """
        broadcast_shape_tensor = torch.tensor(
            broadcast_shape,
            dtype=torch.int32,
        )

        indexes = self._compute_indexes(broadcast_shape_tensor)

        strings = torch.reshape(strings, [-1])

        outputs = torch.zeros(indexes.size())

        for index, string in enumerate(strings):
            outputs[index] = torch.reshape(
                torch.tensor(
                    self._decoder.decode_with_indexes(
                        string,
                        indexes[index].reshape(-1).to(torch.int32).tolist(),
                        self.cdfs.tolist(),
                        self.cdf_sizes.reshape(-1).to(torch.int32).tolist(),
                        self.cdf_offsets.reshape(-1).to(torch.int32).tolist(),
                    ),
                    device=outputs.device,
                    dtype=outputs.dtype,
                ),
                outputs[index].size(),
            )

        shape = Size(
            torch.cat(
                [
                    torch.tensor(strings.size(), dtype=torch.int32),
                    broadcast_shape_tensor,
                    torch.tensor(self.prior_shape, dtype=torch.int32),
                ],
                dim=0,
            ),
        )

        outputs = torch.reshape(outputs, shape).to(self.prior_dtype)

        if self.quantization_offset is not None:
            outputs += self.quantization_offset

        return outputs

    def forward(self, bottleneck: Tensor):
        """Perturbs a tensor with quantization noise and estimates bits.

        Args:
            bottleneck: the data to be compressed. Must have at least
            ``self.coding_rank`` dimensions, and the innermost dimensions must
            be broadcastable to ``self.prior_shape``.

        Returns:
            a pair, ``perturbed`` and ``bits``, where ``perturbed`` is the
            ``bottleneck`` perturbed with quantization noise and ``bits`` is
            the bit rate. ``bits`` has the same shape as ``bottleneck`` without
            the ``self.coding_rank`` innermost dimensions.
        """
        quantized = self.quantize(bottleneck)

        k = torch.sum(
            self.prior.log_prob(quantized),
            range(-self.coding_rank, 0),
        )

        bits = k / 2

        return quantized, bits

    def _pmf_to_cdf(
        self,
        pmfs: Tensor,
        pmf_offsets: Tensor,
        pmf_sizes: Tensor,
        maximum_pmf_size: int,
    ) -> Tensor:
        size = (len(pmf_sizes), maximum_pmf_size + 2)

        cdfs = torch.zeros(
            size,
            dtype=torch.int32,
            device=pmfs.device,
        )

        for index, pmf in enumerate(pmfs):
            pmf = torch.cat(
                [
                    pmf[: typing.cast(int, pmf_sizes[index])],
                    pmf_offsets[index],
                ],
                dim=0,
            )

            cdf = ncF.pmf_to_quantized_cdf(pmf, self.range_coder_precision)

            cdfs[index, : cdf.size()[0]] = cdf

        return cdfs

    def _compute_indexes(self, broadcast_shape: Tensor) -> Tensor:
        return torch.broadcast_to(
            torch.reshape(
                torch.arange(
                    functools.reduce(lambda x, y: x * y, self.prior_shape, 1),
                    dtype=torch.int32,
                ),
                self.prior_shape,
            ),
            Size(
                torch.cat(
                    [
                        broadcast_shape,
                        torch.tensor(
                            self.prior_shape,
                            dtype=torch.int32,
                        ),
                    ],
                    dim=0,
                )
            ),
        )
