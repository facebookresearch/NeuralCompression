from collections import namedtuple

import torch
from torch import Tensor
from ._message_stack import (
    _message_to_message_stack,
    _push_to_message_stack,
    _pop_from_message_stack,
)
from ._unbounded_index_range_encode import _cdf_to_encode

Codec = namedtuple("Codec", ["pop", "push"])


def _cdf_to_decode(cdf_y, cdf_y_length):
    cdf_y = cdf_y[:cdf_y_length]

    def f(cum_freq):
        return torch.searchsorted(cdf_y, cum_freq, right=True) - 1

    return f


def _codec(encode, decode, precision):
    def pop(message):
        encoded, f = _pop_from_message_stack(message, precision)

        decoded = decode(encoded)

        return f(*encode(decoded)), decoded

    def push(message, symbol):
        return _push_to_message_stack(message, *encode(symbol), precision)

    return Codec(pop, push)


def unbounded_index_range_decode(
    data: Tensor,
    index: Tensor,
    cdf: Tensor,
    cdf_size: Tensor,
    offset: Tensor,
    precision: int,
    overflow_width: int,
) -> Tensor:
    """Range decodes encoded data using an indexed probability table.

    Args:
        data: the values to be encoded.
        index: for each value in ``data``, the corresponding value in ``index``
            determines which row in ``cdf`` should be used to encode the value
            in ``data``. ``index`` also determines which element in ``offset``
            determines the integer interval ``cdf`` applies to. Naturally, the
            elements of ``index`` should be in the half-open interval
            [0, cdf.shape[0]).
        cdf: a 2D tensor where each row contains a CDF.
        cdf_size: a 1D tensor whose length should be the same as the number of
            rows of ``cdf``. The values in ``cdf_size`` denote the length of
            the CDF vector in the corresponding row of ``cdf``.
        offset: all the regular data values associated with :math:`index = i`
            should be in the half-open interval
            :math:`[offset_{i}, offset_{i} + m)`.
        precision: the number of bits for probability quantization, must be
            less than or equal to 16.
        overflow_width: the bit width of the variable-length overflow code,
            must be less than or equal to ``precision``.

    Returns:
        The decoded data.
    """
    message_stack = _message_to_message_stack(data, data.shape)

    decoded = torch.flatten(torch.empty(index.shape))

    index = index.to(torch.int32).flatten()

    overflow_cdf = torch.arange((1 << overflow_width) + 1, dtype=torch.int64)

    overflow_pop, overflow_push = _codec(
        _cdf_to_encode(overflow_cdf),
        _cdf_to_decode(overflow_cdf, len(overflow_cdf)),
        overflow_width,
    )

    for i in range(len(index)):
        cdf_index = index[i]
        maximum = cdf_size[cdf_index] - 2

        pop, push = _codec(
            _cdf_to_encode(cdf[cdf_index]),
            _cdf_to_decode(cdf[cdf_index], cdf_size[cdf_index]),
            precision,
        )

        message_stack, v = pop(message_stack)

        if v == maximum:
            message_stack, _v = overflow_pop(message_stack)

            _v = int(_v)

            widths = _v

            while _v == (1 << overflow_width) - 1:
                message_stack, _v = overflow_pop(message_stack)

                _v = int(_v)

                widths += _v

            overflow = 0

            for j in range(widths):
                message_stack, _v = overflow_pop(message_stack)

                _v = int(_v)

                overflow |= _v << (j * overflow_width)

            v = overflow >> 1

            if overflow & 1:
                v = -v - 1
            else:
                v += maximum

        decoded[i] = v + offset[cdf_index]

    return decoded
