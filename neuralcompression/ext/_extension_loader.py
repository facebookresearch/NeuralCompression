# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import List

from torch.utils.cpp_extension import load


def load_extension(extension_name: str, extension_folder: Path, file_names: List[str]):
    """Simple extension loading utility."""
    return load(
        name=extension_name,
        sources=[str(extension_folder / file_name) for file_name in file_names],
        verbose=True,
    )
