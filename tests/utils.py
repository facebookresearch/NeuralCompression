# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from pathlib import Path
import numpy as np
from PIL import Image
import torch
from numpy.random import Generator


def create_input(shape):
    x = np.arange(np.product(shape)).reshape(shape)

    return torch.from_numpy(x).to(torch.get_default_dtype())


def create_random_image(shape, rng: Optional[Generator] = None):
    if rng is None:
        rng = np.random.default_rng()
    img = rng.random(shape)
    img = (255.0 / img.max() * (img - img.min())).astype(np.uint8)
    res = img.astype(np.float32) / 255
    return res


def write_image_to_file(img: np.ndarray, file_path: Path):
    img = (img * 255.0).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    pil_img = Image.fromarray(img)
    pil_img.save(file_path)


def rand_im(shape, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return torch.tensor(rng.uniform(size=shape), dtype=torch.get_default_dtype())
