"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import PIL.Image
import pytest

from neuralcompression.data import CLIC2020
from tests.conftest import create_random_image


@pytest.fixture
def data(tmp_path):
    directory = tmp_path.joinpath("clic2020").joinpath("test")

    directory.mkdir(parents=True)

    for index in range(3):
        path = directory.joinpath(f"{index}.png")

        create_random_image(path, (3, 224, 224))

    return CLIC2020(tmp_path, split="test")


class TestCLIC2020:
    def test___getitem__(self, data):
        assert isinstance(data[0], PIL.Image.Image)

    def test___len__(self, data):
        assert len(data) == 3
