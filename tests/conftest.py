# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch


@pytest.fixture(
    scope="session", params=[(2, 2, 64, 256), (3, 5, 128, 64), (5, 3, 256, 256)]
)
def arange_4d_image(request):
    x = torch.arange(torch.prod(torch.tensor(request.param))).reshape(request.param)

    return x.to(torch.get_default_dtype())


@pytest.fixture(
    scope="session", params=[(2, 2, 65, 257), (3, 5, 124, 63), (5, 3, 252, 257)]
)
def arange_4d_image_odd(request):
    x = torch.arange(torch.prod(torch.tensor(request.param))).reshape(request.param)

    return x.to(torch.get_default_dtype())
