# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple, Union

import torch
import torch.nn
from torch import Size, Tensor

from ._hyperprior_autoencoder import HyperpriorAutoencoder


class ScaleHyperpriorAutoencoder(HyperpriorAutoencoder):
    def __init__(
        self,
        network_channels: int = 128,
        compression_channels: int = 192,
        in_channels: int = 3,
        minimum: Union[int, float] = 0.11,
        maximum: Union[int, float] = 256,
        steps: int = 64,
    ):
        super(ScaleHyperpriorAutoencoder, self).__init__(
            network_channels,
            compression_channels,
            in_channels,
            minimum=minimum,
            maximum=maximum,
            steps=steps,
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        y = self.encoder(x)

        z = self.hyper_encoder(y)

        z_hat, z_scores = self.bottleneck(z)

        scales_hat = self.hyper_decoder(z_hat)

        y_hat, y_scores = self.gaussian_conditional(y, scales_hat)

        x_hat = self.decoder(y_hat)

        return x_hat, (y_scores, z_scores)

    def compress(
        self,
        bottleneck: Tensor,
    ) -> Tuple[List[List[str]], Size]:
        """Compresses a ``Tensor`` to bit strings.
        ``bottleneck`` is quantized then compressed using probability tables.
        The quantized ``Tensor`` can be recovered by calling the
        ``decompress()`` method.
        Args:
            bottleneck: the data to be compressed.
        Returns:
            a string for each coding unit.
        """
        y = self.encoder(bottleneck)

        z = self.hyper_encoder(y)

        z_compressed = self.bottleneck.compress(z)

        z_size = z.size()[-2:]

        z_hat = self.bottleneck.decompress(z_compressed, z_size)

        scales_hat = self.hyper_decoder(z_hat)

        indexes = self.gaussian_conditional.build_indexes(scales_hat)

        y_compressed = self.gaussian_conditional.compress(y, indexes.to(torch.int32))

        compressed = [y_compressed, z_compressed]

        return compressed, z_size

    def decompress(
        self,
        strings: List[str],
        broadcast_size: Size,
    ) -> Tensor:
        """Decompresses a ``Tensor``.
        Reconstructs the quantized ``Tensor`` from bit strings produced by the
        ``compress()`` method. It is necessary to provide a part of the output
        shape in ``broadcast_shape``.
        Args:
            strings: the compressed bit strings.
            broadcast_size: the part of the output ``Tensor`` size between the
                shape of ``strings`` on the left and the prior shape on the
                right. This must match the shape of the input passed to
                ``compress()``.
        Returns:
            has the size ``Size([*strings.size(), *broadcast_shape])``.
        """
        z_hat = self.bottleneck.decompress(strings[1], broadcast_size)

        indexes = self.gaussian_conditional.build_indexes(self.hyper_decoder(z_hat)).to(
            torch.int32
        )

        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)

        x_hat = self.decoder(torch.clamp(y_hat, 0, 1))

        return x_hat
