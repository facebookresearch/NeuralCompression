from typing import Optional, Tuple, Union

import torch
import torch.nn

from torch import Tensor
from torch.nn import Module

from ..functional import lower_bound


class NonNegativeParameter(Module):
    """Non-negative parameter as required by generalized divisive normalization
    (GDN) transformations. The parameter is subjected to an invertible
    transformation that slows down the learning rate for small values.

    Args:
        initial_value: the initial value of the kernel. If not provided, its
            ``shape`` must be given, and the initial value of the parameter
            will be undefined.
        minimum: lower bound for the parameter (defaults to ``0.0``).
        offset: offset added to the reparameterization. The parameterization of
            beta or gamma as their square roots lets the training slow down
            when values are close to zero, which is desirable as small values
            in the denominator can lead to a situation where gradient noise on
            beta or gamma leads to extreme amounts of noise in the GDN
            activations. However, without an ``offset``, we would get zero
            gradients if any elements of beta or gamma were exactly zero, and
            thus the training could get stuck. To prevent this, we add this
            small constant. The default value was empirically determined as a
            good starting point. Making it bigger potentially leads to more
            gradient noise on the activations, making it too small may lead to
            numerical precision issues.
        shape: shape of the initial value of the kernel, ignored unless
            ``initial_value`` is ``None``.
        dtype: ``dtype`` of the parameter, if not provided, inferred from
            `initial_value`.
    """

    _pedestal: Tensor

    def __init__(
        self,
        initial_value: Optional[Tensor] = None,
        minimum: float = 0.0,
        offset: float = 2 ** -18,
        shape: Optional[Union[Tuple[int], int]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(NonNegativeParameter, self).__init__()

        self._minimum = minimum

        self._offset = offset

        if initial_value is None:
            if shape is None:
                error = """
                `initial_value` is `None`, `shape` must be specified
                """

                raise ValueError(error)

            self.initial_value = torch.zeros(shape, dtype=dtype)
        else:
            self.initial_value = torch.tensor(initial_value, dtype=dtype)

        pedestal = torch.tensor(
            [self._offset ** 2],
            dtype=self.initial_value.dtype,
        )

        self._bound = (self._minimum + self._offset ** 2) ** 0.5

        self.initial_value = torch.sqrt(
            torch.max(
                self.initial_value + pedestal,
                pedestal,
            ),
        )

        self.register_buffer("_pedestal", pedestal)

    def forward(self, x: Tensor) -> Tensor:
        return lower_bound(x, self._bound) ** 2 - self._pedestal

    def parameterize(self, x: Tensor) -> Tensor:
        return torch.sqrt(
            torch.max(
                x + self._pedestal,
                self._pedestal,
            ),
        )
