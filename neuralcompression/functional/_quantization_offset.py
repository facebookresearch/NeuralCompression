# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor
from torch.distributions import Distribution


def quantization_offset(distribution: Distribution) -> Tensor:
    """Computes a distribution-dependent quantization offset.

    For range coding of continuous random variables, the values need to be
    quantized first. Typically, it is beneficial for compression performance to
    align the centers of the quantization bins such that one of them coincides
    with the mode of the distribution. With offset being the mode of
    the distribution, for instance, this can be achieved simply by computing:

    .. math::
        \\hat{x} = round(x - offset) + offset

    Note:
        If ``distribution.mean`` is undefined, quantized integer values (i.e.
        an offset of zero) are returned. If the offset is in the range
        :math:`[-0.5, 0.5]` the returned value will be combined with a rounding
        quantizer.

    Args:
        distribution: an object representing a continuous-valued random
            variable.

    Returns:
        a tensor broadcastable to shape ``distribution.batch_shape``,
        containing the determined quantization offsets. No gradients are
        permitted to flow through the return value.
    """
    try:
        offset = distribution.mean
    except AttributeError:
        offset = 0

    with torch.no_grad():
        return offset - torch.round(offset)
