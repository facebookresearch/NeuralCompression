"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import random

import pytest
from PIL.Image import Image

from neuralcompression.data import CLIC2020Image
from tests.utils import create_random_image


@pytest.fixture
def data(tmp_path):
    directory = tmp_path.joinpath("clic2020").joinpath("test")

    directory.mkdir(parents=True)

    n = random.choice(range(16))

    for index in range(n):
        path = directory.joinpath(f"{index}.png")

        create_random_image(path, (3, 224, 224))

    return CLIC2020Image(tmp_path, split="test"), n


class TestCLIC2020:
    def test___getitem__(self, data):
        data, _ = data

        assert isinstance(data[0], Image)

    def test___len__(self, data):
        data, n = data

        assert len(data) == n
