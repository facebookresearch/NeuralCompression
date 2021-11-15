"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
from typing import List, NamedTuple, Optional, OrderedDict, Union

import torch
import torch.nn
from compressai.entropy_models import GaussianConditional
from torch import Size, Tensor

from ._analysis_transformation_2d import AnalysisTransformation2D
from ._hyper_analysis_transformation_2d import HyperAnalysisTransformation2D
from ._hyper_synthesis_transformation_2d import HyperSynthesisTransformation2D
from ._prior import Prior
from ._synthesis_transformation_2d import SynthesisTransformation2D


class _CompressReturnType(NamedTuple):
    strings: List[List[str]]
    broadcast_shape: Size


class _ForwardOutputScores(NamedTuple):
    y: Tensor
    z: Tensor


class _ForwardOutput(NamedTuple):
    scores: _ForwardOutputScores
    x_hat: Tensor


class ScaleHyperprior(Prior):
    def __init__(
        self,
        n: int = 128,
        m: int = 192,
        minimum: Union[int, float] = 0.11,
        maximum: Union[int, float] = 256,
        steps: int = 64,
    ):
        super(ScaleHyperprior, self).__init__(n)

        self._n = n
        self._m = m

        self._minimum = math.log(minimum)
        self._maximum = math.log(maximum)

        self._steps = steps

        self._prior = GaussianConditional(None)

        self._encode = AnalysisTransformation2D(self._n, self._m)
        self._decode = SynthesisTransformation2D(self._n, self._m)

        self._hyper_encode = HyperAnalysisTransformation2D(self._n, self._m)
        self._hyper_decode = HyperSynthesisTransformation2D(self._n, self._m)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    @property
    def scales(self) -> Tensor:
        return torch.exp(torch.linspace(self._minimum, self._maximum, self._steps))

    def forward(self, x: Tensor) -> _ForwardOutput:
        y = self._encode(x)

        z_hat, z_scores = self.bottleneck(self._hyper_encode(torch.abs(y)))

        y_hat, y_scores = self._prior(y, self._hyper_decode(z_hat))

        x_hat = self._decode(y_hat)

        scores = _ForwardOutputScores(y_scores, z_scores)

        return _ForwardOutput(scores, x_hat)

    def load_state_dict(
        self,
        state_dict: OrderedDict[str, Tensor],
        strict: bool = True,
    ):
        self._update_registered_buffers(
            self._prior,
            "_prior",
            [
                "_quantized_cdf",
                "_offset",
                "_cdf_length",
                "scale_table",
            ],
            state_dict,
        )

        super(ScaleHyperprior, self).load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict: OrderedDict):
        n = state_dict["_encode.model.0.weight"].size()[0]
        m = state_dict["_encode.model.6.weight"].size()[0]

        net = cls(n, m)
        net.load_state_dict(state_dict)
        return net

    def update(
        self,
        force: bool = False,
        scales: Optional[Tensor] = None,
    ) -> bool:
        if scales is None:
            scales = self.scales

        updated = self._prior.update_scale_table(scales, force=force)

        updated |= super(ScaleHyperprior, self).update(force=force)

        return updated

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

        z = self._hyper_encode(torch.abs(y))

        z_compressed = self.bottleneck.compress(z)

        z_size = z.size()[-2:]

        z_hat = self.bottleneck.decompress(z_compressed, z_size)

        indexes = self._prior.build_indexes(self._hyper_decode(z_hat))

        y_compressed = self._prior.compress(y, indexes.to(torch.int32))

        compressed = [y_compressed, z_compressed]

        return _CompressReturnType(compressed, z_size)

    def decompress(
        self,
        strings: List[str],
        broadcast_shape: Size,
    ) -> Tensor:
        """Decompresses a ``Tensor``.

        Reconstructs the quantized ``Tensor`` from bit strings produced by the
        ``compress()`` method. It is necessary to provide a part of the output
        shape in ``broadcast_shape``.

        Args:
            strings: the compressed bit strings.
            broadcast_shape: the part of the output ``Tensor`` size between the
                shape of ``strings`` on the left and the prior shape on the
                right. This must match the shape of the input passed to
                ``compress()``.

        Returns:
            has the size ``Size([*strings.size(), *broadcast_shape])``.
        """
        z_hat = self.bottleneck.decompress(strings[1], broadcast_shape)

        indexes = self._prior.build_indexes(self._hyper_decode(z_hat)).to(torch.int32)

        y_hat = self._prior.decompress(strings[0], indexes, z_hat.dtype)

        x_hat = self._decode(torch.clamp(y_hat, 0, 1))

        return x_hat
