# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from neuralcompression.loss_fn import MSELPIPSLoss, NormFixLPIPS


@pytest.mark.parametrize("shape", [(3, 3, 32, 32)])
@pytest.mark.parametrize("backbone", ["alex", "vgg"])
@pytest.mark.parametrize("mse_param", [255.0**2, 1.0])
@pytest.mark.parametrize("lpips_param", [1.0, 0.5])
def test_mse_lpips_loss(shape, backbone, mse_param, lpips_param):
    normalize = True
    rng = torch.Generator()
    rng.manual_seed(int(torch.prod(torch.tensor(shape))))

    base_img = torch.rand(size=shape, generator=rng)
    img = torch.rand(size=shape, generator=rng)

    if not normalize:
        base_img = base_img * 2.0 - 1.0
        img = img * 2.0 - 1.0

    loss_fn = MSELPIPSLoss(
        mse_param=mse_param,
        lpips_param=lpips_param,
        backbone=backbone,
        normalize=normalize,
    )
    output = loss_fn(img, base_img)

    lpips_model = NormFixLPIPS(net=backbone).eval()
    for p in lpips_model.parameters():
        p.requires_grad_(False)
    est = (
        mse_param * torch.mean((img - base_img) ** 2)
        + lpips_param * lpips_model(img, base_img, normalize=normalize).mean()
    )

    assert torch.allclose(output, est)
