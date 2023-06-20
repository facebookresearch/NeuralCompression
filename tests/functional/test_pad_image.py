# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch import Tensor

import neuralcompression.functional as ncF


def create_4d_image(shape) -> Tensor:
    x = torch.arange(torch.prod(torch.tensor(shape))).reshape(shape)

    return x.to(torch.get_default_dtype())


@pytest.mark.parametrize("mode", ["reflect", "constant", "replicate", "circular"])
@pytest.mark.parametrize(
    "factor,start_sz,output_sz",
    [
        (64, (2, 2, 64, 257), (2, 2, 64, 320)),
        (32, (3, 5, 124, 63), (3, 5, 128, 64)),
        (16, (1, 3, 252, 257), (1, 3, 256, 272)),
    ],
)
def test_pad_image(mode: str, factor, start_sz, output_sz):
    arange_image = create_4d_image(start_sz)
    output, (h, w) = ncF.pad_image_to_factor(arange_image, factor, mode=mode)

    assert h == arange_image.shape[2]
    assert w == arange_image.shape[3]

    assert tuple(output.shape) == output_sz
