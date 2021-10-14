from typing import Any

import torch
from torch import Tensor
from torch.autograd import Function


def upper_bound(
    x: Tensor,
    bound: float,
    gradient: str = "identity_if_towards",
) -> Tensor:
    """``torch.minimum`` with a gradient for ``x`` > ``bound``.

    This function is semantically equivalent to ``torch.minimum`` except the
        behavior of the gradient with respect to ``x`` for input values that
        reach the ``bound`` depends on the ``gradient`` option:

        * ``disconnected``, the returned gradient is zero for values that reach
            the bound. This behavior is identical to the behavior of
            ``torch.minimum``.

        * ``identity``, the gradient is unconditionally replaced with the
            identity function.

        * ``identity_if_towards``, the gradient is replaced with the identity
            function, but only if applying gradient descent would push the
            values of inputs towards the bound. For gradient values that push
            away from the bound, the returned gradient is still zero.

    In the latter two cases, ``identity`` and ``identity_if_towards``, no
        gradient is returned for ``bound``. Also, the implementation of
        ``identity_if_towards`` assumes the shape of ``x`` is the same as the
        shape of the output (i.e. it won’t work reliably for all possible
        broadcasting scenarios).

    Args:
        x: the input tensor.
        bound: upper bound for ``x``.
        gradient: The dataset split to use. One of
            {``disconnected``, ``identity``, ``identity_if_towards``}.
            Defaults to ``identity_if_towards``.

    Returns:
        the output tensor.
    """
    return _UpperBound.apply(x, bound, gradient)


class _UpperBound(Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        gradients = ("disconnected", "identity", "identity_if_towards")

        if kwargs["gradient"] in gradients:
            ctx.gradient = kwargs["gradient"]
        else:
            raise ValueError

        ctx.mask = kwargs["x"].le(kwargs["bound"])

        return torch.clamp(kwargs["x"], kwargs["bound"])

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_output = grad_outputs[0]

        if ctx.gradient == "identity_if_towards":
            grad_output *= torch.logical_or(ctx.mask, grad_output.gt(0.0))

        if ctx.gradient == "disconnected":
            grad_output *= ctx.mask

        return grad_output.type(grad_output.dtype), None
