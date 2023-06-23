# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from neuralcompression.loss_fn import MSELoss


@pytest.mark.parametrize("mse_param", [255.0**2, 1.0, 0.5])
def test_mse_loss(mse_param, arange_4d_image):
    rng = torch.Generator()
    rng.manual_seed(int(torch.prod(torch.tensor(arange_4d_image.shape))))

    img = torch.randn(size=arange_4d_image.shape, generator=rng)

    loss_fn = MSELoss(mse_param)

    output = loss_fn(img, arange_4d_image)

    est = mse_param * torch.mean((img - arange_4d_image) ** 2)

    assert torch.allclose(output, est)
