"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Optional, Tuple

import torch
import torch.nn

from torch import Tensor

import neuralcompression.functional as ncF


class NonNegativeParameterization(torch.nn.Module):
    initial_value: Optional[Tensor]
    _pedestal: Tensor

    def __init__(
        self,
        initial_value: Optional[Tensor] = None,
        minimum: float = 0.0,
        offset: float = 2 ** -18,
        shape: Optional[Tuple[int]] = None,
    ):
        super(NonNegativeParameterization, self).__init__()

        self.minimum = minimum

        self.offset = offset

        self.bound = (self.minimum + self.offset ** 2) ** 0.5

        if initial_value is None:
            if shape is None:
                error_message = """
                ``initial_value`` is ``None``, ``shape`` must be specified
                """

                raise ValueError(error_message)

            initial_value = torch.zeros(shape, dtype=torch.float)

        self.register_buffer(
            "_pedestal",
            torch.tensor([(self.offset ** 2)], dtype=initial_value.dtype),
        )

        if initial_value is not None:
            self.initial_value = torch.sqrt(
                torch.max(
                    initial_value + self._pedestal,
                    self._pedestal,
                ),
            )

    def forward(self, x: Tensor) -> Tensor:
        return ncF.lower_bound(x, self.bound) ** 2 - self._pedestal
