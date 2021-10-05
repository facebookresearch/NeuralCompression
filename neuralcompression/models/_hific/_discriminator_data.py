"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import dataclasses

import torch


@dataclasses.dataclass
class _DiscriminatorData:
    authentic_image: torch.Tensor
    synthetic_image: torch.Tensor
    authentic_predictions: torch.Tensor
    synthetic_predictions: torch.Tensor
