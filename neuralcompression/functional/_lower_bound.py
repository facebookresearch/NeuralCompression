import torch
from torch import Tensor
from torch.autograd import Function


class _LowerBound(Function):
    @staticmethod
    def forward(
        ctx,
        tensor: Tensor,
        lower_bound: int,
        gradient: str = "identity_if_towards",
    ) -> Tensor:
        if gradient in ("disconnected", "identity", "identity_if_towards"):
            ctx.gradient = gradient
        else:
            raise ValueError

        ctx.mask = tensor.ge(lower_bound)

        return torch.clamp(tensor, lower_bound)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        if ctx.gradient == "identity_if_towards":
            grad_output *= torch.logical_or(ctx.mask, grad_output.lt(0.0))

        if ctx.gradient == "disconnected":
            grad_output *= ctx.mask

        return grad_output.type(grad_output.dtype), None


lower_bound = _LowerBound.apply
