"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os.path
from pathlib import Path
from typing import Callable, Optional, Tuple
import gzip
import shutil

import PIL
import h5py
import torchvision.datasets.utils
from torchvision.datasets import VisionDataset


class PCam(VisionDataset):
    url = "https://zenodo.org/record/2546921/files"

    def __init__(
        self,
        root: str,
        split: str = "train",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super(PCam, self).__init__(
            root,
            transforms,
            transform,
            target_transform,
        )

        self.root = Path(root)

        self.split = split

        if download:
            self.download()

        self.data = []

        with h5py.File(f"{self.root}/{self._data_path}", "r") as fp:
            for data in fp["x"]:
                self.data += [data]

        self.targets = []

        with h5py.File(f"{self.root}/{self._target_path}", "r") as fp:
            for target in fp["y"]:
                self.targets += [target.flatten()[0]]

    def __getitem__(self, index: int) -> Tuple[PIL.Image.Image, int]:
        data, target = self.data[index], self.targets[index]

        data = PIL.Image.fromarray(data)

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def _data_path(self) -> str:
        return f"camelyonpatch_level_2_split_{self.split}_x.h5"

    @property
    def _target_path(self) -> str:
        return f"camelyonpatch_level_2_split_{self.split}_y.h5"

    def download(self):
        torchvision.datasets.utils.download_url(
            f"{self.url}/{self._data_path}.gz",
            str(self.root),
        )

        if not os.path.exists(f"{self.root}/{self._data_path}.gz"):
            with gzip.open(f"{self.root}/{self._data_path}.gz", "rb") as source:
                with open(f"{self.root}/{self._data_path}", "wb") as destination:
                    shutil.copyfileobj(
                        source,
                        destination,
                    )

        torchvision.datasets.utils.download_url(
            f"{self.url}/{self._target_path}.gz",
            str(self.root),
        )

        if not os.path.exists(f"{self.root}/{self._target_path}.gz"):
            with gzip.open(f"{self.root}/{self._target_path}.gz", "rb") as source:
                with open(f"{self.root}/{self._target_path}", "wb") as destination:
                    shutil.copyfileobj(
                        source,
                        destination,
                    )
