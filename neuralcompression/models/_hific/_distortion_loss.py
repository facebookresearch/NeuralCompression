"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn


def _distortion_loss(authentic: torch.Tensor, synthetic: torch.Tensor) -> torch.Tensor:
    return torch.mean(
        torch.nn.MSELoss(reduction="none")(synthetic * 255.0, authentic * 255.0)
    )
