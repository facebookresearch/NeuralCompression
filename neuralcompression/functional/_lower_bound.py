from typing import Any

import torch
from torch.autograd import Function


class LowerBound(Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        gradients = ("disconnected", "identity", "identity_if_towards")

        if kwargs["gradient"] in gradients:
            ctx.gradient = kwargs["gradient"]
        else:
            raise ValueError

        ctx.mask = kwargs["tensor"].ge(kwargs["lower_bound"])

        return torch.clamp(kwargs["tensor"], kwargs["lower_bound"])

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_output = grad_outputs[0]

        if ctx.gradient == "identity_if_towards":
            grad_output *= torch.logical_or(ctx.mask, grad_output.lt(0.0))

        if ctx.gradient == "disconnected":
            grad_output *= ctx.mask

        return grad_output.type(grad_output.dtype), None
