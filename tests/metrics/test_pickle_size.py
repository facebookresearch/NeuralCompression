# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from neuralcompression.metrics import pickle_size_of


@pytest.mark.parametrize("arr_size", [64, 128, 256, (64, 64)])
def test_pickle_size(arr_size, tmp_path: Path):
    x = np.reshape(np.arange(np.product(arr_size)), arr_size)

    obj = {"thearr": x}

    mem_size = pickle_size_of(obj)

    tmp_file = f"pickle_size_of_{np.product(arr_size)}.pkl"
    with open(tmp_path / tmp_file, "wb") as f:
        pickle.dump(obj, f)

    file_size = os.stat(tmp_path / tmp_file).st_size

    assert mem_size == file_size
