"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import dataclasses

import torch


@dataclasses.dataclass
class _IntermediateData:
    authentic_image: torch.Tensor
    synthetic_image: torch.Tensor
    quantized_latent_features: torch.Tensor
    nbpp: float
    qbpp: float
