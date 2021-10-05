"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from os import rename
from os.path import join
from pathlib import Path
from shutil import rmtree
from typing import Callable, Optional, Union

from PIL.Image import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import (
    download_and_extract_archive,
    verify_str_arg,
)


class CLIC2020(Dataset):
    """`Challenge on Learned Image Compression (CLIC) 2020
    <http://compression.cc/tasks/>`_ Dataset.

    Args:
        root: Root directory where images are downloaded to.
            Expects the following folder structure if download=False:

            .. code::

                <root>
                    └── clic2020
                        ├── test
                        │   ├── *.png
                        ├── train
                        │   ├── *.png
                        └── val
                            └── *.png
        split: The dataset split to use. One of
            {``train``, ``val``, ``test``}.
            Defaults to ``train``.
        download: If true, downloads the dataset from the
            internet and puts it in root directory. If dataset is already
            downloaded, it is not downloaded again.
        transform: A function/transform that takes in a
            PIL image and returns a transformed version.  E.g,
            ``transforms.RandomCrop``.
    """

    splits = {
        "train": {
            "filename": "train.zip",
            "md5": "a6845cac88c3dd882246575f7a2fc5f9",
            "url": join(
                "https://data.vision.ee.ethz.ch/cvl/clic",
                "professional_train_2020.zip",
            ),
        },
        "val": {
            "filename": "val.zip",
            "md5": "7111ee240435911db04dbc5f40d50272",
            "url": join(
                "https://data.vision.ee.ethz.ch/cvl/clic",
                "professional_valid_2020.zip",
            ),
        },
        "test": {
            "filename": "test.zip",
            "md5": "4476b708ea4c492505dd70061bebe202",
            "url": join(
                "https://storage.googleapis.com/clic2021_public",
                "professional_test_2021.zip",
            ),
        },
    }

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        download: bool = False,
        transform: Optional[Callable[[Image], Tensor]] = None,
    ):
        self.root: Path = Path(root).joinpath("clic2020")

        self.split = verify_str_arg(split, "split", ("train", "val", "test"))

        self.transform = transform

        if download:
            self.download()

        self.paths = [*self.root.joinpath(self.split).glob("*.png")]

    def __getitem__(self, index: int) -> Image:
        path = self.paths[index]

        image = default_loader(path)

        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self) -> int:
        return len(self.paths)

    def download(self):
        extract_root = str(self.root)

        if self.split == "test":
            extract_root = self.root.joinpath("test")

        download_and_extract_archive(
            **self.splits[self.split],
            download_root=str(self.root),
            extract_root=extract_root,
        )

        if self.split == "val":
            rename(self.root.joinpath("valid"), self.root.joinpath(self.split))

        if self.split in {"train", "val"}:
            rmtree(self.root.joinpath("__MACOSX"))
        else:
            rmtree(self.root.joinpath(self.split).joinpath("__MACOSX"))
