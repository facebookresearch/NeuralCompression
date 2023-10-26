# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch
import torch.nn as nn


class MockBackbone(nn.Module):
    def __init__(self):
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


@pytest.fixture(scope="session")
def mock_backbone():
    return MockBackbone()
