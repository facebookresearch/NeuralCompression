# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch
import torch.nn as nn
from torch import Tensor


class MockBackbone(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 2048, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        for param in self.parameters():
            param.requires_grad_(False)

        self.eval()

    def train(self, mode: bool = True) -> "MockBackbone":
        """keep network in evaluation mode."""
        return super().train(False)

    def forward(self, image: Tensor) -> Tensor:
        return self.model(image.float() / 255.0).flatten(1)


class MockDiffBackbone(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = MockBackbone()

    def forward(self, image1: Tensor, image2: Tensor) -> Tensor:
        return self.model(image1) - self.model(image2)


@pytest.fixture(
    scope="session", params=[(2, 2, 64, 64), (3, 5, 128, 64), (1, 3, 256, 256)]
)
def arange_4d_image(request):
    x = torch.arange(torch.prod(torch.tensor(request.param))).reshape(request.param)

    return x.to(torch.get_default_dtype())


@pytest.fixture(
    scope="session", params=[(2, 2, 65, 257), (3, 5, 124, 63), (1, 3, 252, 257)]
)
def arange_4d_image_odd(request):
    x = torch.arange(torch.prod(torch.tensor(request.param))).reshape(request.param)

    return x.to(torch.get_default_dtype())
