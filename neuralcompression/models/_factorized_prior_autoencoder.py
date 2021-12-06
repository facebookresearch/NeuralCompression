# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, OrderedDict, Tuple

import torch
from torch import Size, Tensor

from ._prior_autoencoder import PriorAutoencoder


class FactorizedPriorAutoencoder(PriorAutoencoder):
    def __init__(
        self,
        network_channels: int = 128,
        compression_channels: int = 192,
        in_channels: int = 3,
    ):
        super(FactorizedPriorAutoencoder, self).__init__(
            network_channels,
            compression_channels,
            in_channels,
        )

        self.hyper_encoder = None
        self.hyper_decoder = None

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """
        Args:
            x:

        Returns:
        """
        y = self.encoder(x)

        y_hat, y_probabilities = self.bottleneck(y)

        return self.decoder(y_hat), [y_probabilities]

    @classmethod
    def from_state_dict(cls, state_dict: OrderedDict[str, Tensor]):
        """
        Args:
            state_dict:

        Returns:
        """
        n = state_dict["encoder.encode.0.weight"].size()[0]
        m = state_dict["encoder.encode.6.weight"].size()[0]

        prior = cls(n, m)

        prior.load_state_dict(state_dict)

        return prior

    def compress(self, bottleneck: Tensor) -> Tuple[List[List[str]], Size]:
        """
        Args:
            bottleneck:

        Returns:
        """
        y = self.encoder(bottleneck)

        return [self.bottleneck.compress(y)], Size(y.size()[-2:])

    def decompress(self, strings: List[List[str]], broadcast_size: Size) -> Tensor:
        """
        Args:
            strings:
            broadcast_size:

        Returns:
        """
        return torch.clamp(
            self.decoder(
                self.bottleneck.decompress(
                    strings[0],
                    broadcast_size,
                ),
            ),
            min=0,
            max=1,
        )
