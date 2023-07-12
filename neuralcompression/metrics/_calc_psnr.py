# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import numpy as np
import torch
from torch import Tensor


def calc_psnr(image: Tensor, reference: Tensor, data_range: float = 255.0) -> Tensor:
    """
    Calculates PSNR for PyTorch tensors.

    Args:
        image: A 4-D PyTorch tensor.
        reference: A reference 4-D PyTorch tensor.
        data_range: Range of the data. Default is 255.0.

    Returns:
        PSNR between image and reference.
    """
    mse = torch.mean((image - reference) ** 2)
    return 20.0 * math.log10(data_range) - 10.0 * torch.log10(mse)


def calc_psnr_numpy(
    image: np.ndarray, reference: np.ndarray, data_range: float = 255.0
) -> np.ndarray:
    """
    Calculates PSNR for Numpy tensors.

    Args:
        image: A 4-D PyTorch tensor.
        reference: A reference 4-D PyTorch tensor.
        data_range: Range of the data. Default is 255.0.

    Returns:
        PSNR between image and reference.
    """
    mse = np.mean((image - reference) ** 2)
    return 20.0 * math.log10(data_range) - 10.0 * np.log10(mse)
