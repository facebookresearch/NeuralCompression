# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, OrderedDict, Tuple

import torch
from torch import Size, Tensor

from ._prior_autoencoder import PriorAutoencoder


class FactorizedPriorAutoencoder(PriorAutoencoder):
    """Factorized prior autoencoder described in:

        | End-to-end Optimized Image Compression
        | Johannes Ballé, Valero Laparra, Eero P. Simoncelli
        | https://arxiv.org/abs/1611.01704

    Args:
        network_channels: number of channels in the network.
        compression_channels: number of inferred latent compression features.
        in_channels: number of channels in the input image.
    """

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
        y = self.encoder(x)

        y_hat, y_probabilities = self.bottleneck(y)

        return self.decoder(y_hat), [y_probabilities]

    @classmethod
    def from_state_dict(cls, state_dict: OrderedDict[str, Tensor]):
        n = state_dict["encoder.encode.0.weight"].size()[0]
        m = state_dict["encoder.encode.6.weight"].size()[0]

        prior = cls(n, m)

        prior.load_state_dict(state_dict)

        return prior

    def compress(self, bottleneck: Tensor) -> Tuple[List[List[str]], Size]:
        """Compresses a floating-point tensor.

        Compresses the tensor to bit strings. ``bottleneck`` is first quantized
        and then compressed using the probability tables in
        ``self.bottleneck._quantized_cdf`` derived from ``self.bottleneck``.
        The quantized tensor can later be recovered by calling
        ``decompress()``.

        Note:
            The innermost coding rank dimensions are treated as one coding unit
            (i.e. compressed into one string each). Any additional dimensions
            to the left are treated as batch dimensions.

        Args:
            bottleneck: the data to be compressed.

        Returns:
            the compressed data.
        """
        y = self.encoder(bottleneck)

        return [self.bottleneck.compress(y)], Size(y.size()[-2:])

    def decompress(
        self,
        strings: List[List[str]],
        broadcast_size: Size,
    ) -> Tensor:
        """Decompresses a tensor.

        Reconstructs the quantized tensor from bit strings produced by
        ``compress()``. It is necessary to provide a part of the output shape
        in ``broadcast_size``.

        Args:
            strings: the compressed bit strings.
            broadcast_size: the part of the output tensor shape between the
                shape of ``strings`` on the left and the prior shape on the
                right. This must match the shape of the input to
                ``compress()``.

        Returns:
            the decompressed data.
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
