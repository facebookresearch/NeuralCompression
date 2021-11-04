"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Callable, List, Optional, Tuple

import PIL
import torchvision.datasets.utils
from torchvision.datasets import VisionDataset


class Food101(VisionDataset):
    _classes: List[str] = []
    _paths: List[str] = []

    def __init__(
        self,
        root: str,
        split: str = "train",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super(Food101, self).__init__(
            root,
            transforms,
            transform,
            target_transform,
        )

        self.root = root

        self.split = split

        self.transform = transform

        self.target_transform = target_transform

        if download:
            self.download()

        with open("./food-101/meta/classes.txt", "r") as fp:
            for category in fp.readlines():
                self._classes += category.split()

        with open(f"{self.root}/meta/{self.split}.txt") as fp:
            for name in fp.readlines():
                self._paths += name.split()

    def __getitem__(self, index: int) -> Tuple[PIL.Image.Image, int]:
        path = f"{self.root}/images/{self._paths[index]}.jpg"

        data = torchvision.datasets.folder.default_loader(path)

        if self.transform is not None:
            data = self.transform(data)

        directory, _ = self._paths[index].split("/")

        target = self._classes.index(directory)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(self) -> int:
        return len(self._paths)

    def download(self):
        torchvision.datasets.utils.download_and_extract_archive(
            download_root=self.root,
            extract_root=".",
            remove_finished=True,
            url="https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz",
        )
