# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Callable, Optional

import torch
import torch.nn.functional
from torch import Tensor
from torch.nn import Module, Parameter

from ._non_negative_parameterization import NonNegativeParameterization


class GeneralizedDivisiveNormalization(Module):
    """Applies generalized divisive normalization for each channel across a
    batch of data.

    Implements an activation function that is a multivariate generalization of
    the following sigmoid-like function:

    .. math::
        y_{i}=\\frac{x_{i}}{(\\beta_{i}+\\sum_{j}\\gamma_{ij}|x_{j}|^{\\alpha_{ij}})^{\\epsilon_{i}}}

    where :math:`i` and :math:`j` map over channels.

    This implementation never sums across spatial dimensions. It is similar to
    local response normalization, but much more flexible, as :math:`\\alpha`,
    :math:`\\beta`, :math:`\\gamma`, and :math:`\\epsilon` are trainable
    parameters.

    The method was originally described in:

        | “Density Modeling of Images using a Generalized Normalization
            Transformation”
        | Johannes Ballé, Valero Laparra, Eero P. Simoncelli
        | https://arxiv.org/abs/1511.06281

    and expanded in:

        | “End-to-end Optimized Image Compression”
        | Johannes Ballé, Valero Laparra, Eero P. Simoncelli
        | https://arxiv.org/abs/1611.01704

    Args:
        channels: number of channels in the input.
        inverse: compute the generalized divisive normalization response. If
            ``True``, compute the inverse generalized divisive normalization
            response (one step of fixed point iteration to invert the
            generalized divisive normalization; the division is replaced by
            multiplication).
        alpha_parameter: A ``Tensor`` means that the value of ``alpha`` is
            fixed. ``None`` means that when the layer is initialized, a
            ``NonNegativeParameterization`` layer is created to train ``alpha``
            (with a minimum value of ``1``). The default is a fixed value of
            ``1``.
        beta_parameter: A ``Tensor`` means that the value of ``beta`` is fixed.
            ``None`` means that when the layer is initialized, a
            ``NonNegativeParameterization`` layer is created to train ``beta``
            (with a minimum value of ``1e-6``).
        epsilon_parameter: A ``Tensor`` means that the value of ``epsilon`` is
            fixed. ``None`` means that when the layer is initialized, a
            ``NonNegativeParameterization`` layer is created to train
            ``epsilon`` (with a minimum value of 1e-6). The default is a fixed
            value of ``1``.
        gamma_parameter: A ``Tensor`` means that the value of ``gamma`` is
            fixed. ``None`` means that when the layer is initialized, a
            ``NonNegativeParameterization`` layer is created to train
            ``gamma``.
        alpha_initializer: initializes the ``alpha`` parameter. Only used if
            ``alpha`` is trained. Defaults to ``1``.
        beta_initializer: initializes the ``beta`` parameter. Only used if
            ``beta`` is created when initializing the layer. Defaults to ``1``.
        epsilon_initializer: initializes the ``epsilon`` parameter. Only used
            if ``epsilon`` is trained. Defaults to ``1``.
        gamma_initializer: initializes the ``gamma`` parameter. Only used if
            ``gamma`` is created when initializing the layer. Defaults to the
            identity multiplied by ``0.1``. A good default value for the
            diagonal is somewhere between ``0`` and ``0.5``. If set to ``0``
            and ``beta`` is initialized as ``1``, the layer is effectively
            initialized to the identity operation.
    """

    alpha: Parameter
    beta: Parameter
    epsilon: Parameter
    gamma: Parameter

    def __init__(
        self,
        channels: int,
        inverse: bool = False,
        alpha_parameter: Optional[Tensor] = None,
        beta_parameter: Optional[Tensor] = None,
        epsilon_parameter: Optional[Tensor] = None,
        gamma_parameter: Optional[Tensor] = None,
        alpha_initializer: Optional[Callable[[Tensor], Tensor]] = None,
        beta_initializer: Optional[Callable[[Tensor], Tensor]] = None,
        epsilon_initializer: Optional[Callable[[Tensor], Tensor]] = None,
        gamma_initializer: Optional[Callable[[Tensor], Tensor]] = None,
    ):
        super(GeneralizedDivisiveNormalization, self).__init__()

        self._channels = torch.tensor(channels, dtype=torch.int32)

        self._inverse = inverse

        if alpha_parameter is None:
            if alpha_initializer is None:
                alpha_initializer = functools.partial(
                    lambda x: torch.ones(x),
                )

            self._reparameterized_alpha = NonNegativeParameterization(
                alpha_initializer(self._channels),
                minimum=1,
            )

            if self._reparameterized_alpha.initial_value is not None:
                self.alpha = Parameter(
                    self._reparameterized_alpha.initial_value,
                )
        else:
            if isinstance(alpha_parameter, Parameter):
                self.alpha = alpha_parameter
            else:
                alpha_parameter = torch.tensor(alpha_parameter)

                self.alpha = Parameter(alpha_parameter)

        if beta_parameter is None:
            if beta_initializer is None:
                beta_initializer = functools.partial(
                    lambda x: torch.ones(x),
                )

            self._reparameterized_beta = NonNegativeParameterization(
                beta_initializer(self._channels),
                minimum=1e-6,
            )

            if self._reparameterized_beta.initial_value is not None:
                self.beta = Parameter(
                    self._reparameterized_beta.initial_value,
                )
        else:
            if isinstance(beta_parameter, Parameter):
                self.beta = beta_parameter
            else:
                beta_parameter = torch.tensor(beta_parameter)

                self.beta = Parameter(beta_parameter)

        if epsilon_parameter is None:
            if epsilon_initializer is None:
                epsilon_initializer = functools.partial(
                    lambda x: torch.ones(x),
                )

            self._reparameterized_epsilon = NonNegativeParameterization(
                epsilon_initializer(self._channels),
                minimum=1e-6,
            )

            if self._reparameterized_epsilon.initial_value is not None:
                self.epsilon = Parameter(
                    self._reparameterized_epsilon.initial_value,
                )
        else:
            if isinstance(epsilon_parameter, Parameter):
                self.epsilon = epsilon_parameter
            else:
                epsilon_parameter = torch.tensor(epsilon_parameter)

                self.epsilon = Parameter(epsilon_parameter)

        if gamma_parameter is None:
            if gamma_initializer is None:
                gamma_initializer = functools.partial(
                    lambda x: 0.1 * torch.eye(x),
                )

            self._reparameterized_gamma = NonNegativeParameterization(
                gamma_initializer(self._channels),
                minimum=0,
            )

            if self._reparameterized_gamma.initial_value is not None:
                self.gamma = Parameter(
                    self._reparameterized_gamma.initial_value,
                )
        else:
            if isinstance(gamma_parameter, Parameter):
                self.gamma = gamma_parameter
            else:
                gamma_parameter = torch.tensor(gamma_parameter)

                self.gamma = Parameter(gamma_parameter)

    def forward(self, x: Tensor) -> Tensor:
        _, channels, _, _ = x.size()

        y = torch.nn.functional.conv2d(
            x ** 2,
            torch.reshape(
                self._reparameterized_gamma(self.gamma),
                (channels, channels, 1, 1),
            ),
            self._reparameterized_beta(self.beta),
        )

        if self._inverse:
            return x * torch.sqrt(y)

        return x * torch.rsqrt(y)
