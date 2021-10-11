"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import Tensor


def pmf_to_quantized_cdf(pmf: Tensor, precision: int) -> Tensor:
    """Transforms a probability mass function (PMF) into a quantized cumulative
        distribution function (CDF) for entropy coding.

    Because this op uses floating-point operations the quantized output may not
        be consistent across multiple platforms. For entropy encoders and
        decoders to have the same quantized CDF on different platforms, the
        quantized CDF should be transformed, saved, and used on the multiple
        platforms.

    After quantization, if the PMF does not sum to :math:`2^{precision}`, then
        some values of PMF are increased or decreased until the sum is equal to
        :math:`2^{precision}`.

    Args:
        pmf: a probability mass function (PMF).
        The PMF is **not** normalized by this op. The user is responsible for
            normalizing the PMF, if necessary.
        precision: the number of bits for probability quantization, must be
            less than or equal to 16.

    Returns:
        The quantized CDF.
    """
    cdf = torch.zeros(pmf.shape[0] + 1)

    cdf[1:] = torch.cumsum(pmf, dim=0)

    cdf = torch.round(cdf * (1 << precision) / cdf[-1]).to(torch.int64)

    for j in range(cdf.size(0) - 1):
        if cdf[j] == cdf[j + 1]:
            peak = (1 << precision) + 1

            best = -1

            for k in range(cdf.size(0) - 1):
                frequency = int(cdf[k + 1] - cdf[k])

                if 1 < frequency < peak:
                    peak = frequency

                    best = k

            assert best != -1

            if best < j:
                for k in range(best + 1, j + 1):
                    cdf[k] -= 1
            else:
                assert best > j

                for k in range(j + 1, best + 1):
                    cdf[k] += 1

    assert cdf[0] == 0

    assert cdf[-1] == 1 << precision

    return cdf
