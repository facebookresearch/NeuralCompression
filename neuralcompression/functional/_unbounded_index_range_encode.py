from typing import Callable, List, Tuple

from torch import Tensor, arange, int32, int64


def _cdf_to_f(
    cdf_y: Tensor,
) -> Callable[[int], Tuple[Tensor, Tensor]]:
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
    overflow_width: int = 4,
) -> List[Tuple[Tensor, Tensor, bool]]:
    instructions = []

    data = data.to(int32).flatten()
    index = index.to(int32).flatten()

    f = _cdf_to_f(arange((1 << overflow_width) + 1, dtype=int64))

    for i in range(len(index)):
        cdf_index = index[i]

        cdf_y = cdf[cdf_index]

        maximum = cdf_size[cdf_index] - 2

        v = data[i] - offset[cdf_index]

        if v < 0:
            overflow = -2 * v - 1

            v = maximum
        elif v >= maximum:
            overflow = 2 * (v - maximum)

            v = maximum
        else:
            overflow = 0

        instructions += [(*_cdf_to_f(cdf_y)(v), False)]

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

    return instructions
