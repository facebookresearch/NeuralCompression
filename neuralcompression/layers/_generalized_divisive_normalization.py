"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import functools
from typing import Callable, Optional, Union

import torch
import torch.nn.functional
from torch import Tensor
from torch.nn import Module, Parameter

from ._non_negative_parameterization import NonNegativeParameterization


class GeneralizedDivisiveNormalization(Module):
    """Generalized divisive normalization

    Implements an activation function that is a multivariate generalization of
    the following sigmoid-like function:

        y[i] = x[i] / (beta[i] + sum_j(gamma[j, i] * |x[j]|^alpha))^epsilon

    where ``i`` and ``j`` map over channels. This implementation never sums
    across spatial dimensions. It is similar to local response normalization,
    but much more flexible, as ``alpha``, ``beta``, ``gamma``, and ``epsilon``
    are trainable parameters.

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
        rectify: If ``True``, apply a ``ReLU`` non-linearity to the inputs
            before calculating the generalized divisive normalization response.
        alpha_parameter: A ``Tensor`` means that the value of ``alpha`` is
            fixed. A ``Callable`` can be used to determine the value of
            ``alpha`` as a function of some other parameter or tensor. This can
            be a ``Parameter``. ``None`` means that when the layer is
            initialized, a ``NonNegativeParameterization`` layer is created to
            train ``alpha`` (with a minimum value of ``1``). The default is a
            fixed value of ``1``.
        beta_parameter: A ``Tensor`` means that the value of ``beta`` is fixed.
            A ``Callable`` can be used to determine the value of ``beta`` as a
            function of some other parameter or tensor. This can be a
            ``Parameter``. ``None`` means that when the layer is initialized, a
            ``NonNegativeParameterization`` layer is created to train ``beta``
            (with a minimum value of ``1e-6``).
        epsilon_parameter: A ``Tensor`` means that the value of ``epsilon`` is
            fixed. A ``Callable`` can be used to determine the value of
            ``epsilon`` as a function of some other parameter or tensor. This
            can be a ``Parameter``. ``None`` means that when the layer is
            initialized, a ``NonNegativeParameterization`` layer is created to
            train ``epsilon`` (with a minimum value of 1e-6). The default is a
            fixed value of ``1``.
        gamma_parameter: A ``Tensor`` means that the value of ``gamma`` is
            fixed. A ``Callable`` can be used to determine the value of
            ``gamma`` as a function of some other parameter or tensor. This can
            be a ``Parameter``. ``None`` means that when the layer is
            initialized, a ``NonNegativeParameterization`` layer is created to
            train ``gamma``.
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

    def __init__(
        self,
        channels: Union[int, Tensor],
        inverse: bool = False,
        rectify: bool = False,
        alpha_parameter: Union[float, int, Tensor, Parameter] = None,
        beta_parameter: Union[float, int, Tensor, Parameter] = None,
        epsilon_parameter: Union[float, int, Tensor, Parameter] = None,
        gamma_parameter: Union[float, int, Tensor, Parameter] = None,
        alpha_initializer: Optional[Callable[[Tensor], Tensor]] = None,
        beta_initializer: Optional[Callable[[Tensor], Tensor]] = None,
        epsilon_initializer: Optional[Callable[[Tensor], Tensor]] = None,
        gamma_initializer: Optional[Callable[[Tensor], Tensor]] = None,
    ):
        super(GeneralizedDivisiveNormalization, self).__init__()

        self._inverse = inverse

        self._rectify = rectify

        if alpha_parameter is None:
            if alpha_initializer is None:
                alpha_initializer = torch.ones

            self._reparameterized_alpha = NonNegativeParameterization(
                alpha_initializer(channels),
                minimum=1,
            )

            self._alpha_parameter = Parameter(
                self._reparameterized_alpha.initial_value,
            )
        else:
            self._alpha_parameter = alpha_parameter

        if beta_parameter is None:
            if beta_initializer is None:
                beta_initializer = torch.ones

            self._reparameterized_beta = NonNegativeParameterization(
                beta_initializer(channels),
                minimum=1e-6,
            )

            self._beta_parameter = Parameter(
                self._reparameterized_beta.initial_value,
            )
        else:
            self._beta_parameter = beta_parameter

        if epsilon_parameter is None:
            if epsilon_initializer is None:
                epsilon_initializer = torch.ones

            self._reparameterized_epsilon = NonNegativeParameterization(
                epsilon_initializer(channels),
                minimum=1e-6,
            )

            self._epsilon_parameter = Parameter(
                self._reparameterized_epsilon.initial_value,
            )
        else:
            self._epsilon_parameter = epsilon_parameter

        if gamma_parameter is None:
            if gamma_initializer is None:
                gamma_initializer = functools.partial(lambda x: 0.1 * torch.eye(x))

            self._reparameterized_gamma = NonNegativeParameterization(
                gamma_initializer(channels),
                minimum=0,
            )

            self._gamma_parameter = Parameter(
                self._reparameterized_gamma.initial_value,
            )
        else:
            self._gamma_parameter = gamma_parameter

    def forward(self, x: Tensor) -> Tensor:
        _, channels, _, _ = x.size()

        if self._rectify:
            x = torch.nn.functional.relu(x)

        y = torch.nn.functional.conv2d(
            x ** 2,
            torch.reshape(
                self._reparameterized_gamma(self._gamma_parameter),
                (channels, channels, 1, 1),
            ),
            self._reparameterized_beta(self._beta_parameter),
        )

        if self._inverse:
            return x * torch.sqrt(y)

        return x * torch.rsqrt(y)
