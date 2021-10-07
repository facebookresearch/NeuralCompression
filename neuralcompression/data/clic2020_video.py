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
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets.utils import verify_str_arg
from tqdm import tqdm


class CLIC2020Video(Dataset):
    """`Challenge on Learned Image Compression (CLIC) 2020
    <http://compression.cc/tasks/>`_ Video Dataset.

    Args:
        root: Root directory where images are downloaded to.
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
        max_workers: Optional[int] = None,
    ):
        self.root = Path(root).joinpath("clic-2020-video")

        self.root.mkdir(exist_ok=True, parents=True)

        self.split = verify_str_arg(split, "split", ("train", "val", "test"))

        self.val_frames = self._create_data_dictionary(self.VAL_FRAMES_FILE)
        self.test_frames = self._create_data_dictionary(self.TEST_FRAMES_FILE)

        self.transform = transform

        self.max_workers = max_workers

        if download:
            self.download()

    def __getitem__(self, index: int) -> Image:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def download(self):
        for split in ("train", "val", "test"):
            self.root.joinpath(split).mkdir(exist_ok=True, parents=True)

        with urlopen(f"{self.URL}/video_urls.txt") as file:
            endpoints = file.read().decode("utf-8").splitlines()

        # FIXME: remove after testing
        endpoints = endpoints[:8]

        def f(endpoint: str):
            sleep(0.001)

            path, _ = urlretrieve(endpoint)

            with ZipFile(path, "r") as archive:
                archive.extractall(self.root.joinpath("train"))

        with tqdm(total=len(endpoints)) as progress:
            with ThreadPoolExecutor(self.max_workers) as executor:
                futures = {
                    executor.submit(f, endpoint): endpoint for endpoint in endpoints
                }

                completed = {}

                for future in as_completed(futures):
                    endpoint = futures[future]

                    completed[endpoint] = future.result()

                    progress.update()

    def _create_data_dictionary(self, file: str) -> Dict:
        with urlopen(f"{self.URL}/{file}") as f:
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
