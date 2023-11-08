# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch import Tensor
from torchmetrics.functional.image import peak_signal_noise_ratio

from neuralcompression.metrics import calc_psnr, calc_psnr_numpy


@pytest.mark.parametrize("data_range", [255.0, 1.0])
def test_calc_psnr_torch(data_range: float, arange_4d_image: Tensor):
    reference = torch.randn_like(arange_4d_image)

    val1 = calc_psnr(arange_4d_image, reference, data_range=data_range)
    val2 = peak_signal_noise_ratio(arange_4d_image, reference, data_range=data_range)

    assert torch.allclose(val1, val2)


@pytest.mark.parametrize("data_range", [255.0, 1.0])
def test_calc_psnr_numpy(data_range: float, arange_4d_image: Tensor):
    reference = torch.randn_like(arange_4d_image)

    val2 = peak_signal_noise_ratio(arange_4d_image, reference, data_range=data_range)

    val1 = torch.tensor(
        calc_psnr_numpy(
            arange_4d_image.numpy(), reference.numpy(), data_range=data_range
        )
    ).float()

    assert torch.allclose(val1, val2)
