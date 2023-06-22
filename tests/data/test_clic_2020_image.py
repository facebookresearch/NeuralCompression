# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy
import pytest
from PIL.Image import Image
from utils import create_random_image, write_image_to_file

from neuralcompression.data import CLIC2020Image


@pytest.fixture
def data(tmp_path):
    rng = numpy.random.default_rng(0xFEEEFEEE)

    directory = tmp_path.joinpath("clic2020").joinpath("test")

    directory.mkdir(parents=True)

    n = int(rng.integers(1, 16, (1,)))

    for index in range(n):
        path = directory.joinpath(f"{index}.png")

        img = create_random_image((3, 224, 224), rng)
        write_image_to_file(img, path)

    return CLIC2020Image(tmp_path, split="test"), n


class TestCLIC2020Image:
    def test___getitem__(self, data):
        data, _ = data

        assert isinstance(data[0], Image)
        assert isinstance(data[-1], Image)

    def test___len__(self, data):
        data, n = data

        assert len(data) == n
