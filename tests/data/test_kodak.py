# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms as tv_transforms

from neuralcompression.data import Kodak


def create_random_image(file_path, shape):
    img = np.random.random_sample(shape)
    img = (255.0 / img.max() * (img - img.min())).astype(np.uint8)
    res = torch.tensor(img).to(torch.float) / 255
    img = np.transpose(img, (1, 2, 0))
    img = Image.fromarray(img)
    img.save(file_path)
    return res


@pytest.mark.parametrize("img_sizes, num_imgs", [([(3, 512, 768), (3, 768, 512)], 5)])
def test_kodak_image_dataset(img_sizes, num_imgs, tmp_path, monkeypatch):
    def _check_integrity_mock(a, b):
        return True

    monkeypatch.setattr(Kodak, "_check_integrity", _check_integrity_mock)

    kodak_root = tmp_path / "kodak"
    kodak_root.mkdir()

    test_imgs = []
    for i in range(num_imgs):
        size = img_sizes[np.random.randint(len(img_sizes))]
        test_imgs.append(create_random_image(f"{kodak_root}/kodim{i+1:02d}.png", size))

    kodak_dataset = Kodak(
        kodak_root,
        transform=tv_transforms.Compose([tv_transforms.ToTensor()]),
    )

    for i in range(num_imgs):
        assert torch.isclose(test_imgs[i], kodak_dataset[i]).all()
