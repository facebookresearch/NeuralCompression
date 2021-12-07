# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
import torch.nn
from torch import Tensor

import neuralcompression.functional as ncF


class NonNegativeParameterization(torch.nn.Module):
    """Non-negative parameterization.

    The parameter is subjected to an invertible transformation that slows down
    the learning rate for small values.

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
        lower_bound_gradient: The gradient to use for the ``lower_bound``
            operation. One of {``disconnected``, ``identity``,
            ``identity_if_towards``}. Defaults to ``identity_if_towards``.
    """

    initial_value: Optional[Tensor]
    _pedestal: Tensor

    def __init__(
        self,
        initial_value: Optional[Tensor] = None,
        minimum: float = 0.0,
        offset: float = 2 ** -18,
        shape: Optional[Tuple[int]] = None,
        lower_bound_gradient: str = "identity_if_towards",
    ):
        super(NonNegativeParameterization, self).__init__()

        self._minimum = minimum

        self._offset = offset

        self._lower_bound_gradient = lower_bound_gradient

        self._bound = (self._minimum + self._offset ** 2) ** 0.5

        if initial_value is None:
            if shape is None:
                error_message = """
                ``initial_value`` is ``None``, ``shape`` must be specified
                """

                raise ValueError(error_message)

            initial_value = torch.zeros(shape, dtype=torch.float)

        self.register_buffer(
            "_pedestal",
            torch.tensor([(self._offset ** 2)], dtype=initial_value.dtype),
        )

        if initial_value is not None:
            self.initial_value = torch.sqrt(
                torch.max(
                    initial_value + self._pedestal,
                    self._pedestal,
                ),
            )

    def forward(self, x: Tensor) -> Tensor:
        lower_bound = ncF.lower_bound(
            x,
            self._bound,
            self._lower_bound_gradient,
        )

        return lower_bound ** 2 - self._pedestal
