"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import Tensor


class _LowerBound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        x, bound, gradient = args

        gradients = ("disconnected", "identity", "identity_if_towards")

        if gradient not in gradients:
            raise ValueError

        ctx.save_for_backward(torch.ge(x, bound))

        ctx.bound, ctx.gradient = bound, gradient

        return torch.clamp_max(x, bound)

    @staticmethod
    def backward(ctx, *gradient_outputs):
        (y,) = gradient_outputs

        (x,) = ctx.saved_tensors

        if ctx.gradient == "disconnected":
            z = x
        elif ctx.gradient == "identity":
            z = y
        elif ctx.gradient == "identity_if_towards":
            z = torch.logical_or(x, y.lt(0.0))
        else:
            raise ValueError

        return (y * z).type(y.dtype), None, None


def lower_bound(
    x: Tensor,
    bound: float,
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
        gradient: The dataset split to use. One of
            {``disconnected``, ``identity``, ``identity_if_towards``}.
            Defaults to ``identity_if_towards``.

    Returns:
        the output tensor.
    """
    return _LowerBound.apply(x, bound, gradient)
