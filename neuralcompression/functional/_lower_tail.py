# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import enum
from enum import IntEnum
from typing import Union

import torch
import torch.nn
from torch import Tensor
from torch.autograd import Function


@enum.unique
class _LowerBoundGradient(IntEnum):
    disconnected = 0
    identity = 1
    identity_if_towards = 2


class _LowerBound(Function):
    @staticmethod
    def backward(ctx, *grad_outputs):
        (y,) = grad_outputs

        x, bound, gradient = ctx.saved_tensors

        if int(gradient) == _LowerBoundGradient.disconnected:
            return x, None, None

        if int(gradient) == _LowerBoundGradient.identity:
            return y, None, None

        return (((x >= bound) | (y < 0)) * y), None, None

    @staticmethod
    def forward(ctx, *args):
        x, bound, gradient = args

        bound = torch.tensor(bound, dtype=x.dtype)

        ctx.save_for_backward(
            x,
            bound,
            torch.tensor(gradient, dtype=torch.uint8),
        )

        return torch.max(x, bound)


def lower_bound(
    x: Tensor,
    bound: Union[float, Tensor],
    gradient: str = "identity_if_towards",
) -> Tensor:
    """``torch.maximum`` with a gradient for :math:`x < bound`.

    This function is semantically equivalent to ``torch.maximum`` except the
    behavior of the gradient with respect to :math:`x` for input values that
    reach the :math:`bound` depends on the ``gradient`` option:

        * ``"disconnected"``, the returned gradient is zero for values that
          reach the :math:`bound`. This behavior is identical to the behavior
          of ``torch.maximum``.

        * ``"identity"``, the gradient is unconditionally replaced with the
          identity function.

        * ``"identity_if_towards"``, the gradient is replaced with the identity
          function, but only if applying gradient descent would push the values
          of inputs towards the :math:`bound`. For gradient values that push
          away from the :math:`bound`, the returned gradient is still zero.

    In the latter two cases, ``"identity"`` and ``"identity_if_towards"``, no
    gradient is returned for :math:`bound`. Also, the implementation of
    ``"identity_if_towards"`` assumes the shape of :math:`x` is the same as the
    shape of the output (i.e. it won’t work reliably for all possible
    broadcasting scenarios).

    Args:
        x: the input tensor.
        bound: upper bound for :math:`x`.
        gradient: The gradient to use. One of {``"disconnected"``,
            ``"identity"``, ``"identity_if_towards"``}. Defaults to
            ``"identity_if_towards"``.

    Returns:
        the output tensor.
    """

    if gradient not in ("disconnected", "identity", "identity_if_towards"):
        raise ValueError

    return _LowerBound.apply(
        x,
        bound,
        _LowerBoundGradient[gradient].value,
    )
