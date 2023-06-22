# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import numpy
import pytest
import torch
from PIL.Image import Image
from utils import create_random_image, write_image_to_file

from neuralcompression.data import DIV2KDataset
from torchvision.transforms import ToTensor


@pytest.fixture
def div2k_datafolder(tmp_path: Path):
    rng = numpy.random.default_rng(0xFEEEFEEE)

    directory = tmp_path.joinpath("div2k").joinpath("DIV2K_valid_HR")

    directory.mkdir(parents=True)

    n = int(rng.integers(1, 16, (1,)))

    images = []

    for index in range(n):
        path = directory.joinpath(f"{index:03}.png")

        images.append(create_random_image((3, 224, 224), rng))
        write_image_to_file(images[-1], path)

    return directory.parent, images


def test_div2k_getitem(div2k_datafolder):
    transform = ToTensor()
    dataset = DIV2KDataset(div2k_datafolder[0], transform=transform)

    true_images = div2k_datafolder[1]
    for img, true_img in zip(dataset, true_images):
        assert torch.allclose(img, torch.tensor(true_img))
