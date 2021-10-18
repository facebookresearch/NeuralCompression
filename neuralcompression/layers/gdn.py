"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import neuralcompression.functional as ncF


class SimplifiedGDN(nn.Module):
    r"""
    Simplified Generalized Divisive Normalization (GDN).

    This applies the GDN layer, has been shown to be more useful for
    compression than other activation layers such as relu or tanh.
    Mathematically, it applies:

    .. math::
        z_i = \frac{x_i}{\beta_i + \sum_j \gamma_{ij}|x_j|}

    where :math:`\beta_i` and :math:`\gamma_{ij}` are trainable parameters.

    Johnston N, Eban E, Gordon A, Ballé J. Computationally efficient neural
    image compression. arXiv preprint arXiv:1912.08771. 2019 Dec 18.

    Args:
        channels: Number of channels in input.
        gamma_init: Initial value for ``gamma``.
        beta_min: Threshold for ``beta`` for numerical stability.
    """

    def __init__(self, channels: int, gamma_init: float = 0.1, beta_min: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(
            torch.eye(channels).view(channels, channels, 1, 1) * gamma_init
        )
        self.beta = nn.Parameter(torch.ones(channels))
        self.beta_min = beta_min

    def forward(self, image: Tensor) -> Tensor:
        r"""
        Apply forward GDN layer.

        Args:
            image: Input image of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

        Returns:
            ``image`` after normalization.
        """
        # threshold for numerical stability while keeping backprop
        self.gamma.data = ncF.lower_bound(self.gamma.data, 0)
        self.beta.data = ncF.lower_bound(self.beta.data, self.beta_min)

        return image / F.conv2d(torch.abs(image), self.gamma, self.beta)


class SimplifiedInverseGDN(nn.Module):
    r"""
    Simplified Inverse Generalized Divisive Normalization (GDN).

    This applies the inverse GDN layer, has been shown to be more useful for
    compression than other activation layers such as relu or tanh.
    Mathematically, it applies:

    .. math::
        z_i = x_i * {\beta_i + \sum_j \gamma_{ij}|x_j|}

    where :math:`\beta_i` and :math:`\gamma_{ij}` are trainable parameters.

    Johnston N, Eban E, Gordon A, Ballé J. Computationally efficient neural
    image compression. arXiv preprint arXiv:1912.08771. 2019 Dec 18.

    Args:
        channels: Number of channels in input.
        gamma_init: Initial value for ``gamma``.
        beta_min: Threshold for ``beta`` for numerical stability.
    """

    def __init__(self, channels: int, gamma_init: float = 0.1, beta_min: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(
            torch.eye(channels).view(channels, channels, 1, 1) * gamma_init
        )
        self.beta = nn.Parameter(torch.ones(channels))
        self.beta_min = beta_min

    def forward(self, image: Tensor) -> Tensor:
        r"""
        Apply inverse GDN layer.

        Args:
            image: Input image of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

        Returns:
            ``image`` after inverse normalization.
        """
        # threshold for numerical stability while keeping backprop
        self.gamma.data = ncF.lower_bound(self.gamma.data, 0)
        self.beta.data = ncF.lower_bound(self.beta.data, self.beta_min)

        return image * F.conv2d(torch.abs(image), self.gamma, self.beta)
