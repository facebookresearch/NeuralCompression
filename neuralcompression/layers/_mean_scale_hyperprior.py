"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import List

import torch
from torch import Size, Tensor
from torch.nn import Conv2d, ConvTranspose2d, LeakyReLU, Sequential

from ._hyper_analysis_transformation_2d import HyperAnalysisTransformation2D
from ._scale_hyperprior import (
    ScaleHyperprior,
    _CompressReturnType,
    _ForwardReturnType,
    _ForwardReturnTypeScores,
)


class MeanScaleHyperprior(ScaleHyperprior):
    def __init__(self, n: int = 128, m: int = 192, **kwargs):
        super(MeanScaleHyperprior, self).__init__(n, m, **kwargs)

        activation = LeakyReLU(inplace=True)

        self._hyper_encode = HyperAnalysisTransformation2D(n, m, activation)

        self._hyper_decode = Sequential(
            ConvTranspose2d(n, m, (5, 5), (2, 2), (2, 2), (1, 1)),
            activation,
            ConvTranspose2d(m, m * 3 // 2, (5, 5), (2, 2), (2, 2), (1, 1)),
            activation,
            Conv2d(m * 3 // 2, m * 2, (3, 3), (1, 1), 1),
        )

    def forward(self, x: Tensor) -> _ForwardReturnType:
        y = self._encode(x)

        z_hat, z_scores = self.bottleneck(self._hyper_encode(y))

        scales, offsets = self._hyper_decode(z_hat).chunk(2, 1)

        y_hat, y_scores = self._prior(y, scales, offsets)

        scores = _ForwardReturnTypeScores(y_scores, z_scores)

        x_hat = self._decode(y_hat)

        return _ForwardReturnType(scores, x_hat)

    def compress(self, bottleneck: Tensor) -> _CompressReturnType:
        """Compresses a ``Tensor`` to bit strings.

        ``bottleneck`` is quantized then compressed using probability tables.
        The quantized ``Tensor`` can be recovered by calling the
        ``decompress()`` method.

        Args:
            bottleneck: the data to be compressed.

        Returns:
            a string for each coding unit.
        """
        y = self._encode(bottleneck)

        z = self._hyper_encode(y)

        z_strings = self.bottleneck.compress(z)

        size = z.size()[-2:]

        z_hat = self.bottleneck.decompress(z_strings, size)

        scales, offsets = torch.chunk(self._hyper_decode(z_hat), 2, 1)

        indexes = self._prior.build_indexes(scales)

        y_strings = self._prior.compress(y, indexes, offsets)

        strings = [y_strings, z_strings]

        return _CompressReturnType(strings, size)

    def decompress(self, strings: List[str], broadcast_size: Size) -> Tensor:
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
        assert isinstance(strings, list) and len(strings) == 2

        z_hat = self.bottleneck.decompress(strings[1], broadcast_size)

        scales, offsets = torch.chunk(self._hyper_decode(z_hat), 2, 1)

        indexes = self._prior.build_indexes(scales)

        y_hat = self._prior.decompress(strings[0], indexes, offsets)

        x_hat = torch.clamp(self._decode(y_hat), 0, 1)

        return x_hat
