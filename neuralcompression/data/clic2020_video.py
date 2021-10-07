"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import re
import json
import collections
import typing
import urllib.request
import os.path
from pathlib import Path
from typing import Callable, Optional, Union

from PIL.Image import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets.utils import verify_str_arg
import tqdm
import zipfile


class CLIC2020Video(Dataset):
    """`Challenge on Learned Image Compression (CLIC) 2020
    <http://compression.cc/tasks/>`_ Video Dataset.

    Args:
        root: Root directory where images are downloaded to.
            Expects the following folder structure if ``download=False``:

            .. code::

                <root>
                    └── clic-2020-video
                        ├── test
                        │   ├── [A-Za-z]_[720|1080|2160]P-[0-9a-z]{4}
                        │   │   └── [A-Za-z]_[720|1080|2160]P-[0-9a-z]{4}_[0-9]{5}_[yuv].png
                        ├── train
                        │   ├── [A-Za-z]_[720|1080|2160]P-[0-9a-z]{4}
                        │   │   └── [A-Za-z]_[720|1080|2160]P-[0-9a-z]{4}_[0-9]{5}_[yuv].png
                        └── val
                            └── [A-Za-z]_[720|1080|2160]P-[0-9a-z]{4}
                                └── [A-Za-z]_[720|1080|2160]P-[0-9a-z]{4}_[0-9]{5}_[yuv].png
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
    URL = "https://storage.googleapis.com/clic2021_public/txt_files"

    URLS_FILE = "video_urls.txt"
    VAL_FRAMES_FILE = "video_targets_valid.txt"
    TEST_FRAMES_FILE = "video_targets_test.txt"

    def __init__(
            self,
            root: Union[str, Path],
            split: str = "train",
            download: bool = False,
            transform: Optional[Callable[[Image], Tensor]] = None,
    ):
        self.root = Path(root).joinpath("clic2020").joinpath("video")

        self.root.mkdir(exist_ok=True, parents=True)

        self.split = verify_str_arg(split, "split", ("train", "val", "test"))

        self.val_frames = self._create_data_dictionary(self.VAL_FRAMES_FILE)
        self.test_frames = self._create_data_dictionary(self.TEST_FRAMES_FILE)

        self.transform = transform

        if download:
            self.download()

    def __getitem__(self, index: int) -> Image:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def download(self):
        for split in ("train", "val", "test"):
            self.root.joinpath(split).mkdir(exist_ok=True, parents=True)

        with urllib.request.urlopen(f"{self.URL}/video_urls.txt") as file:
            endpoints = file.read().decode("utf-8").splitlines()

        for endpoint in tqdm.tqdm(endpoints[:2]):
            path, _ = urllib.request.urlretrieve(endpoint)

            with zipfile.ZipFile(path, "r") as file:
                file.extractall(self.root.joinpath("train"))

    def _create_data_dictionary(self, file: str) -> typing.Dict:
        pattern = re.compile("(.+_.+-.+)_(.+)_[yuv].png")

        with urllib.request.urlopen(f"{self.URL}/{file}") as f:
            names = f.read().decode("utf-8").splitlines()

        frames = []

        for name in names:
            if name.endswith(".png"):
                frames += [[*re.findall(pattern, name)[0]]]

        dictionary = collections.defaultdict(list)

        for k, *v in frames:
            dictionary[k].append(v)

        for k in dictionary:
            dictionary[k] = sorted([*{*sum(dictionary[k], [])}])

        return dict(dictionary)


if __name__ == "__main__":
    clic2020_video = CLIC2020Video(
        "/Users/allengoodman/Documents/com/github/0x00b1/NeuralCompression",
        download=True
    )
