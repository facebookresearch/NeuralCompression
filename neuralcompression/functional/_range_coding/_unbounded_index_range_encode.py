from typing import Callable, Tuple

import torch
from torch import Tensor

from ._message_stack import (
    _empty_message_stack,
    _message_stack_to_message,
    _push_to_message_stack,
)


def _cdf_to_encode(cdf_y: Tensor) -> Callable[[int], Tuple[Tensor, Tensor]]:
    def f(x: int) -> Tuple[Tensor, Tensor]:
        return cdf_y[x], cdf_y[int(x + 1)] - cdf_y[x]

    return f


def unbounded_index_range_encode(
    data: Tensor,
    index: Tensor,
    cdf: Tensor,
    cdf_size: Tensor,
    offset: Tensor,
    precision: int,
    overflow_width: int,
) -> Tensor:
    """Range encodes unbounded integer data using an indexed probability table.

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
        A range-coded scalar string.
    """
    instructions = []

    data = data.to(torch.int32).flatten()
    index = index.to(torch.int32).flatten()

    f = _cdf_to_encode(torch.arange((1 << overflow_width) + 1, dtype=torch.int64))

    for i in range(len(index)):
        cdf_index = index[i]

        cdf_y = cdf[cdf_index]

        maximum = int(cdf_size[cdf_index] - 2)

        v = int(data[i] - offset[cdf_index])

        if v < 0:
            overflow = -2 * v - 1

            v = maximum
        elif v >= maximum:
            overflow = 2 * (v - maximum)

            v = maximum
        else:
            overflow = 0

        instructions += [(*_cdf_to_encode(cdf_y)(v), False)]

        if v == maximum:
            widths = 0

            while (overflow >> (widths * overflow_width)) != 0:
                widths += 1

            v = widths

            while v >= (1 << overflow_width) - 1:
                instructions += [(*f((1 << overflow_width) - 1), True)]

                v -= (1 << overflow_width) - 1

            instructions += [(*f(v), True)]

            for j in range(widths):
                v = (overflow >> (j * overflow_width)) & (1 << overflow_width) - 1

                instructions += [(*f(v), True)]

    message_stack = _empty_message_stack(())

    for instructions_index in reversed(range(len(instructions))):
        start, frequency, overflowed = instructions[instructions_index]

        if overflowed:
            message_stack = _push_to_message_stack(
                message_stack, start, frequency, overflow_width
            )
        else:
            message_stack = _push_to_message_stack(
                message_stack, start, frequency, precision
            )

    return _message_stack_to_message(message_stack)
