# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple, Union

import torch
from torch import Size, Tensor
from torch.nn import Conv2d, ConvTranspose2d, LeakyReLU, Module, Sequential

from neuralcompression.layers import HyperAnalysisTransformation2D

from ._hyperprior_autoencoder import HyperpriorAutoencoder


class _HyperSynthesisTransformation2D(Module):
    def __init__(
        self,
        network_channels: int = 128,
        compression_channels: int = 192,
        in_channels: int = 3,
    ):
        super(_HyperSynthesisTransformation2D, self).__init__()

        self.decode = Sequential(
            ConvTranspose2d(
                network_channels,
                compression_channels,
                (5, 5),
                (2, 2),
                (5 // 2, 5 // 2),
                (2 - 1, 2 - 1),
            ),
            LeakyReLU(inplace=True),
            ConvTranspose2d(
                compression_channels,
                compression_channels * 3 // 2,
                (5, 5),
                (2, 2),
                (5 // 2, 5 // 2),
                (2 - 1, 2 - 1),
            ),
            LeakyReLU(inplace=True),
            Conv2d(
                compression_channels * in_channels // 2,
                compression_channels * 2,
                (in_channels, in_channels),
                (1, 1),
                (in_channels // 2, in_channels // 2),
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(x)


class MeanScaleHyperpriorAutoencoder(HyperpriorAutoencoder):
    """Mean scale hyperprior autoencoder described in:

        | “Variational Image Compression with a Scale Hyperprior”
        | Johannes Ballé, David Minnen, Saurabh Singh, Sung Jin Hwang,
            Nick Johnston
        | https://arxiv.org/abs/1802.01436

    Args:
        network_channels: number of channels in the network.
        compression_channels: number of inferred latent compression features.
        in_channels: number of channels in the input image.
        minimum: minimum scale size.
        maximum: maximum scale size.
        steps: number of scale steps.
    """

    def __init__(
        self,
        network_channels: int = 128,
        compression_channels: int = 192,
        in_channels: int = 3,
        minimum: Union[int, float] = 0.11,
        maximum: Union[int, float] = 256,
        steps: int = 64,
    ):
        super(MeanScaleHyperpriorAutoencoder, self).__init__(
            network_channels,
            compression_channels,
            in_channels,
            None,
            None,
            HyperAnalysisTransformation2D(
                network_channels,
                compression_channels,
                in_channels,
                LeakyReLU(inplace=True),
            ),
            _HyperSynthesisTransformation2D(
                network_channels,
                compression_channels,
                in_channels,
            ),
            minimum,
            maximum,
            steps,
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        y = self.encoder(x)

        z = self.hyper_encoder(y)

        z_hat, z_scores = self.bottleneck(z)

        scales, offsets = self.hyper_decoder(z_hat).chunk(2, 1)

        y_hat, y_scores = self.gaussian_conditional(y, scales, offsets)

        x_hat = self.decoder(y_hat)

        return x_hat, (y_scores, z_scores)

    def compress(
        self,
        bottleneck: Tensor,
    ) -> Tuple[List[List[str]], Size]:
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
        y = self.encoder.forward(bottleneck)

        z = self.hyper_encoder(y)

        z_strings = self.bottleneck.compress(z)

        size = z.size()[-2:]

        z_hat = self.bottleneck.decompress(z_strings, size)

        gaussian_params = self.hyper_decoder(z_hat)

        scales, offsets = torch.chunk(gaussian_params, 2, 1)

        indexes = self.gaussian_conditional.build_indexes(scales)

        y_strings = self.gaussian_conditional.compress(y, indexes, offsets)

        return [y_strings, z_strings], size

    def decompress(
        self,
        strings: List[str],
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
        assert isinstance(strings, list) and len(strings) == 2

        z_hat = self.bottleneck.decompress(strings[1], broadcast_size)

        gaussian_params = self.hyper_decoder(z_hat)

        scales, offsets = torch.chunk(gaussian_params, 2, 1)

        indexes = self.gaussian_conditional.build_indexes(scales)

        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, offsets)

        x_hat = torch.clamp(self.decoder(y_hat), 0, 1)

        return x_hat
