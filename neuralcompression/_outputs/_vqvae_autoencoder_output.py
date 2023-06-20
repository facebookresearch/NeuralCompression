# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional, Sequence, Union

from torch import Tensor


@dataclass
class VqVaeAutoencoderOutput:
    image: Optional[Tensor] = None
    latent: Optional[Tensor] = None
    prequantized_latent: Optional[Union[Tensor, Sequence[Tensor]]] = None
    commitment_loss: Optional[Tensor] = None
    embedding_loss: Optional[Tensor] = None
    codebook_indices: Optional[Union[Tensor, Sequence[Tensor]]] = None
    quantize_residuals: Optional[Tensor] = None
    num_bytes: Optional[int] = None
    quantize_distances: Optional[Tensor] = None
    indices: Optional[Tensor] = None
