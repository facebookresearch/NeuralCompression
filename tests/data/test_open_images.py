# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import numpy
import pytest
import torch
from torchvision.transforms import ToTensor
from utils import create_deterministic_image, write_image_to_file

from neuralcompression.data import OpenImagesV6


@pytest.fixture
def open_images_datafolder(tmp_path: Path):
    rng = numpy.random.default_rng(0xFEEEFEEE)

    root = tmp_path / "open_images"

    def write_folder_and_flist(base_dir: Path, subfolder: str):
        directory = base_dir / subfolder
        directory.mkdir(parents=True)

        n = int(rng.integers(1, 16, (1,)))

        images = []

        with open(base_dir / f"list-{subfolder}-files.txt", "w") as f:
            for index in range(n):
                path = directory.joinpath(f"{index:03}.jpg")

                images.append(create_deterministic_image((3, 224, 224), index))
                write_image_to_file(images[-1], path)
                f.write(f"{path.name}\n")

            return images

    train_images = write_folder_and_flist(root, "train")
    val_images = write_folder_and_flist(root, "validation")
    test_images = write_folder_and_flist(root, "test")

    return root, [train_images, val_images, test_images]


def test_div2k_getitem(open_images_datafolder):
    transform = ToTensor()

    for split, true_images in zip(["train", "val", "test"], open_images_datafolder[1]):
        dataset = OpenImagesV6(
            open_images_datafolder[0], split=split, transform=transform
        )

        for img, true_img in zip(dataset, true_images):
            # because these are saved with JPEG, we need a really high tol
            assert torch.allclose(img, torch.tensor(true_img), rtol=0.01, atol=0.01)

        assert len(dataset) == len(true_images)
