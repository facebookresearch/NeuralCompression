# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor

import neuralcompression.ext


def pmf_to_quantized_cdf(pmf: Tensor, precision: int) -> Tensor:
    """Transforms a probability mass function (PMF) into a quantized cumulative
    distribution function (CDF) for entropy coding.

    Note:
        Because this function uses floating-point operations the quantized
        output may not be consistent across multiple platforms. For entropy
        encoders and decoders to have the same quantized CDF on different
        platforms, the quantized CDF should be transformed, saved, and used on
        the multiple platforms.

    Note:
        After quantization, if the PMF does not sum to :math:`2^{precision}`,
        then some values of PMF are increased or decreased until the sum is
        equal to :math:`2^{precision}`.

    Args:
        pmf: a probability mass function (PMF).
            The PMF is **not** normalized by this function. The user is
            responsible for normalizing the PMF, if necessary.
        precision: the number of bits for probability quantization, must be
            greater than 0 and less than or equal to 32.

    Returns:
        the quantized CDF.
    """
    if torch.all(pmf <= 0.0).item():
        message = """
        `pmf` has a non-finite or negative member. Please check for numerical
        problems in the probability computation."
        """

        raise ValueError(message)

    if pmf.shape[-1] < 2:
        message = """
        `pmf` shape should be at least two in last axis.
        """

        raise ValueError(message)

    if precision < 1 or precision > 32:
        message = """
        `precision` must be greater than 0 and less than or equal to 32.
        """

        raise ValueError(message)

    cdf = neuralcompression.ext.pmf_to_quantized_cdf(pmf.tolist(), precision)

    return torch.tensor(cdf)
