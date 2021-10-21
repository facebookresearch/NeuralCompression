"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
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
        (grad_output,) = grad_outputs

        (x,) = ctx.saved_tensors

        gradient = int(ctx.gradient)

        if gradient == _LowerBoundGradient.disconnected:
            return x

        if gradient == _LowerBoundGradient.identity:
            return grad_output

        return ((x >= ctx.bound) | (grad_output < 0)) * grad_output, None, None

    @staticmethod
    def forward(ctx, *args):
        x, bound, gradient = args

        if gradient not in ("disconnected", "identity", "identity_if_towards"):
            raise ValueError

        bound = torch.tensor([bound], dtype=x.dtype)

        ctx.bound = bound

        ctx.gradient = torch.tensor(
            _LowerBoundGradient[gradient].value,
            dtype=torch.uint8,
        )

        ctx.save_for_backward(x)

        return torch.max(x, bound)


def lower_bound(
    x: Tensor,
    bound: Union[float, Tensor],
    gradient: str = "identity_if_towards",
) -> Tensor:
    """``torch.maximum`` with a gradient for ``x`` < ``bound``.

    This function is semantically equivalent to ``torch.maximum`` except the
    behavior of the gradient with respect to ``x`` for input values that reach
    the ``bound`` depends on the ``gradient`` option:

        * ``disconnected``, the returned gradient is zero for values that reach
            the bound. This behavior is identical to the behavior of
            ``torch.maximum``.

        * ``identity``, the gradient is unconditionally replaced with the
            identity function.

        * ``identity_if_towards``, the gradient is replaced with the identity
            function, but only if applying gradient descent would push the
            values of inputs towards the bound. For gradient values that push
            away from the bound, the returned gradient is still zero.

    In the latter two cases, ``identity`` and ``identity_if_towards``, no
    gradient is returned for ``bound``. Also, the implementation of
    ``identity_if_towards`` assumes the shape of ``x`` is the same as the shape
    of the output (i.e. it won’t work reliably for all possible broadcasting
    scenarios).

    Args:
        x: the input tensor.
        bound: upper bound for ``x``.
        gradient: The gradient to use. One of {``disconnected``, ``identity``,
            ``identity_if_towards``}. Defaults to ``identity_if_towards``.

    Returns:
        the output tensor.
    """
    return _LowerBound.apply(x, bound, gradient)
