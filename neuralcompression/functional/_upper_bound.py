from typing import Any

import torch
from torch.autograd import Function


class _UpperBound(Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        gradients = ("disconnected", "identity", "identity_if_towards")

        if kwargs["gradient"] in gradients:
            ctx.gradient = kwargs["gradient"]
        else:
            raise ValueError

        ctx.mask = kwargs["tensor"].le(kwargs["bound"])

        return torch.clamp(kwargs["tensor"], kwargs["bound"])

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_output = grad_outputs[0]

        if ctx.gradient == "identity_if_towards":
            grad_output *= torch.logical_or(ctx.mask, grad_output.gt(0.0))

        if ctx.gradient == "disconnected":
            grad_output *= ctx.mask

        return grad_output.type(grad_output.dtype), None
