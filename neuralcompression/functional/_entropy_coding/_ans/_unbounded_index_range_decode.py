from collections import namedtuple

from torch import Tensor, arange, empty, flatten, int32, int64, searchsorted

from ._message_stack import _pop, _push, _to_message_stack
from ._unbounded_index_range_encode import _cdf_to_encode

Codec = namedtuple("Codec", ["pop", "push"])


def _cdf_to_decode(cdf_y, cdf_y_length):
    cdf_y = cdf_y[:cdf_y_length]

    def f(cum_freq):
        return searchsorted(cdf_y, cum_freq, right=True) - 1

    return f


def _codec(encode, decode, precision):
    def pop(message):
        encoded, f = _pop(message, precision)

        decoded = decode(encoded)

        return f(*encode(decoded)), decoded

    def push(message, symbol):
        return _push(message, *encode(symbol), precision)

    return Codec(pop, push)


def unbounded_index_range_decode(
    data: Tensor,
    index: Tensor,
    cdf: Tensor,
    cdf_size: Tensor,
    offset: Tensor,
    precision: int,
    overflow_width: int,
):
    """Range decodes encoded data using an indexed probability table.

    Args:
        data:
        index:
        cdf:
        cdf_size:
        offset:
        precision:
        overflow_width:

    Returns:
        The decoded data.
    """
    message_stack = _to_message_stack(data)

    decoded = flatten(empty(index.shape))

    index = index.to(int32).flatten()

    overflow_cdf = arange((1 << overflow_width) + 1, dtype=int64)

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
