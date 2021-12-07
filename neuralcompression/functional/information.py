# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from torch import Tensor


def information_content(
    probabilities: Tensor, reduction: str = "sum", base: float = 2.0
) -> Tensor:
    r"""
    Estimate information content given a set of probabilities.

    The information content is estimated using Shannon's source coding theorem,
    i.e., :math:`-\sum_{i}^N log(p_i)`, where :math:`p_i` is the :math:`i`th
    index of ``probabilities``. With logarithm base-2, this function estimates
    the number of bits output by an entropy coder.

    Note:
        The expected value of the information content over a distribution is
        the entropy.

    Args:
        probabilities: A tensor of probability estimates.
        reduction: Type of reduction to apply. If ``"sum"``, applies the
            summation across all batch elements. If ``"batch_el"``, applies
            the sum for each batch element. If ``"none"``, this function simply
            applies the logarithm in the specified base.
        base: Logarithm base.

    Returns
        Estimate of information content.
    """
    if reduction not in ("sum", "batch_el", "none"):
        raise ValueError(
            f"Unrecognized reduction {reduction}, must be 'sum', 'batch_el', or 'none'."
        )

    if reduction == "sum":
        natural = torch.sum(torch.log(probabilities))
    elif reduction == "batch_el":
        natural = torch.log(probabilities).view(probabilities.shape[0], -1).sum(dim=-1)
    elif reduction == "none":
        natural = torch.log(probabilities)

    return (-1 / math.log(base)) * natural
