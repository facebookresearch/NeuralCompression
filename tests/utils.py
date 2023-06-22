# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from numpy.random import Generator
from PIL import Image


def create_input(shape, offset: int = 0):
    x = np.arange(np.product(shape)).reshape(shape) + offset

    return torch.from_numpy(x).to(torch.get_default_dtype())


def create_deterministic_image(shape, offset: int = 0):
    img = create_input(shape, offset).numpy()
    img = ((255.0 / img.max()) * img).astype(np.uint8)
    res = img.astype(np.float32) / 255
    return res


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
    if ".jpg" in file_path.name or ".jpeg" in file_path.name:
        pil_img.save(file_path, quality=100, subsampling=0)
    else:
        pil_img.save(file_path)


def rand_im(shape, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return torch.tensor(rng.uniform(size=shape), dtype=torch.get_default_dtype())
