"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pathlib
import shutil

import pytest

from neuralcompression.data import CLIC2020


@pytest.fixture(autouse=True)
def remove_data_directory():
    yield

    if pathlib.Path("data").exists():
        shutil.rmtree("data")


@pytest.fixture
def test_data():
    return CLIC2020("data", "test")


class TestCLIC2020:
    def test___getitem__(self, test_data):
        with pytest.raises(IndexError):
            assert test_data[0]

    def test___len__(self, test_data):
        assert len(test_data) == 0

    def test_download(self, test_data):
        test_data.download()

        assert len(test_data) == 60
