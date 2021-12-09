# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor

from neuralcompression.data._vimeo_90k_septuplet import Vimeo90kSeptuplet


def setup_dummy_dataset(root_dir: Path):
    # Creates a dummy dataset mirroring the structure of Vimeo-90k,
    # where each image in the dataset is filled with a unique constant value.

    with open(root_dir / "sep_trainlist.txt", "w") as file:
        file.write("001\n003\n")

    with open(root_dir / "sep_testlist.txt", "w") as file:
        file.write("002\n")

    for i in range(1, 4):
        folder = root_dir / f"sequences/00{i}"
        folder.mkdir(parents=True)
        for j in range(1, 8):
            img = Image.fromarray(
                np.ones((10, 10, 3), dtype=np.int8) * (10 * i) + j, "RGB"
            )
            img.save(folder / f"im{j}.png")


@pytest.mark.parametrize(
    "frames_per_group",
    list(range(1, 8)),
)
def test_video_mode(tmp_path: Path, frames_per_group: int):
    setup_dummy_dataset(tmp_path)

    ds = Vimeo90kSeptuplet(
        root=tmp_path,
        as_video=True,
        frames_per_group=frames_per_group,
        split="train",
        pil_transform=Compose([ToTensor(), lambda t: 2 * t]),
        tensor_transform=lambda t: 3 * t,
    )

    assert len(ds) == 2

    assert torch.allclose(
        ds[0],
        torch.ones(frames_per_group, 3, 10, 10)
        * torch.tensor([10 + i for i in range(1, frames_per_group + 1)]).view(
            -1, 1, 1, 1
        )
        * 6
        / 255,
    )

    assert torch.allclose(
        ds[1],
        torch.ones(frames_per_group, 3, 10, 10)
        * torch.tensor([30 + i for i in range(1, frames_per_group + 1)]).view(
            -1, 1, 1, 1
        )
        * 6
        / 255,
    )


@pytest.mark.parametrize(
    "frames_per_group",
    list(range(1, 8)),
)
def test_image_mode(tmp_path: Path, frames_per_group: int):
    setup_dummy_dataset(tmp_path)

    ds = Vimeo90kSeptuplet(
        root=tmp_path,
        as_video=False,
        frames_per_group=frames_per_group,
        split="test",
        pil_transform=ToTensor(),
        tensor_transform=lambda t: 5 * t,
    )

    assert len(ds) == frames_per_group

    for i in range(frames_per_group):
        item = ds[i]
        assert torch.allclose(item, torch.ones(3, 10, 10) * (20 + i + 1) * 5 / 255)
