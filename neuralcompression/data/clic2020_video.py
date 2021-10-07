"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from re import findall
from time import sleep
from typing import Callable, Dict, Optional, Union
from urllib.request import urlopen, urlretrieve
from zipfile import ZipFile

from PIL.Image import Image
from torch import Tensor, stack
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import verify_str_arg
from torchvision.transforms import ToTensor
from tqdm import tqdm


class CLIC2020Video(Dataset):
    """`Challenge on Learned Image Compression (CLIC) 2020
    <http://compression.cc/tasks/>`_ Video Dataset.

    Args:
        root: Root directory where videos are downloaded to.
            Expects the following folder structure if ``download=False``:

            .. code::

                <root>
                    └── [A-Za-z]_[720|1080|2160]P-[0-9a-z]{4}
                        └── [A-Za-z]_[720|1080|2160]P-[0-9a-z]{4}[0-9]{5}_[yuv].png
        split: The dataset split to use. One of
            {``train``, ``val``, ``test``}.
            Defaults to ``train``.
        download: If true, downloads the dataset from the
            internet and puts it in root directory. If dataset is already
            downloaded, it is not downloaded again.
        transform: A function/transform that takes in a
            PIL image and returns a transformed version.  E.g,
            ``transforms.RandomCrop``.
        image_transform: A function/transform that takes in a
            PIL image and returns a transformed version.  E.g,
            ``transforms.RandomCrop``.
    """

    url_root = "https://storage.googleapis.com/clic2021_public/txt_files"

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        download: bool = False,
        transform: Optional[Callable[[Tensor], Tensor]] = None,
        image_transform: Optional[Callable[[Image], Tensor]] = None,
    ):
        self.root = Path(root)

        self.split = verify_str_arg(split, "split", ("train", "val", "test"))

        self.val_frames = self._create_data_dictionary("video_targets_valid.txt")
        self.test_frames = self._create_data_dictionary("video_targets_test.txt")

        self.transform = transform

        if image_transform:
            self.image_transform = image_transform
        else:
            self.image_transform = ToTensor()

        if download:
            self.download()

        self.videos = [Path(path.name) for path in self.root.glob("*")]

    def __getitem__(self, index: int) -> Tensor:
        path = self.root.joinpath(self.videos[index])

        frames = []

        for path in sorted([*path.glob("*_y.png")])[:16]:
            frames += [self.image_transform(default_loader(path))]

        video = stack(frames)

        if self.transform:
            video = self.transform(video)

        return video

    def __len__(self) -> int:
        return len(self.videos)

    def download(self):
        self.root.mkdir(exist_ok=True, parents=True)

        with urlopen(f"{self.url_root}/video_urls.txt") as file:
            endpoints = file.read().decode("utf-8").splitlines()

        def f(endpoint: str):
            sleep(0.001)

            path, _ = urlretrieve(endpoint)

            with ZipFile(path, "r") as archive:
                archive.extractall(self.root)

        with tqdm(total=len(endpoints)) as progress:
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(
                        f,
                        endpoint,
                    ): endpoint
                    for endpoint in endpoints
                }

                completed = {}

                for future in as_completed(futures):
                    endpoint = futures[future]

                    completed[endpoint] = future.result()

                    progress.update()

    def _create_data_dictionary(self, file: str) -> Dict:
        with urlopen(f"{self.url_root}/{file}") as f:
            names = f.read().decode("utf-8").splitlines()

        frames = []

        for name in names:
            if name.endswith(".png"):
                frames += [[*findall("(.+_.+-.+)_(.+)_[yuv].png", name)[0]]]

        dictionary = defaultdict(list)

        for k, *v in frames:
            dictionary[k].append(v)

        for k in dictionary:
            dictionary[k] = sorted([*{*sum(dictionary[k], [])}])

        return dict(dictionary)
