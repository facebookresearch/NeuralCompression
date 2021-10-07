"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from re import findall
from time import sleep
from typing import Callable, Dict, List, Optional, Type, Union
from urllib.request import urlopen, urlretrieve
from zipfile import ZipFile

from PIL.Image import Image
from pytorchvideo.data.clip_sampling import ClipSampler
from pytorchvideo.data.frame_video import FrameVideo
from torch import Tensor, clamp, linspace
from torch.utils.data import Dataset, IterableDataset, RandomSampler, Sampler
from torchvision.datasets.utils import verify_str_arg
from torchvision.transforms import ToTensor
from tqdm import tqdm
from pytorchvideo.data import make_clip_sampler
from pytorchvideo.data.utils import MultiProcessSampler


class CLIC2020Video(IterableDataset):
    """`Challenge on Learned Image Compression (CLIC) 2020
    <http://compression.cc/tasks/>`_ Video Dataset.

    Args:
        root: Root directory where videos are downloaded to.
            Expects the following folder structure if ``download=False``:

            .. code::

                <root>
                    └── [A-Za-z]_[720|1080|2160]P-[0-9a-z]{4}
                        └── *[0-9]{5}_[yuv].png
        split: The dataset split to use.
            One of {``train``, ``val``, ``test``}. Defaults to ``train``.
        download: If true, downloads the dataset from the internet and puts it
            in the root directory. If the dataset is already downloaded, it is
            not downloaded again.
        transform: A callable that transforms a clip, e.g,
            ``pytorchvideo.transforms.RandomResizedCrop``.
        multithreaded_io: If true, performs input/output operations across
            multiple threads.
        image_transform: A function/transform that takes in a
            PIL image and returns a transformed version.  E.g,
            ``transforms.RandomCrop``.
    """

    url_root = "https://storage.googleapis.com/clic2021_public/txt_files"

    def __init__(
            self,
            root: Union[str, Path],
            clip_sampler: ClipSampler,
            split: str = "train",
            download: bool = False,
            transform: Optional[Callable[[Tensor], Tensor]] = None,
            image_transform: Optional[Callable[[Image], Tensor]] = None,
            video_sampler: Type[Sampler] = RandomSampler,
            multithreaded_io: bool = False,
            frames_per_clip: Optional[int] = None,
    ):
        self.root = Path(root)

        self.clip_sampler = clip_sampler

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

        self._video_paths = [Path(path.name) for path in self.root.glob("*")]

        self._video_sampler = video_sampler(self._video_paths)

        self._video_sampler_iterator = None

        if frames_per_clip:
            self.frames_per_clip = frames_per_clip

            self.frame_filter = self._sample_clip_frames

        self._current_video = None
        self._current_video_clip = None
        self._next_clip_start_sec = 0.0

    @property
    def video_sampler(self) -> Sampler:
        return self._video_sampler

    def _sample_clip_frames(self, frame_indices: List[int]) -> List[int]:
        n = len(frame_indices)

        return [
            frame_indices[index]
            for index in (
                clamp(
                    linspace(
                        0,
                        n - 1,
                        self.frames_per_clip,
                    ),
                    0,
                    n - 1,
                ).long()
            )
        ]

    def __next__(self) -> dict:
        if not self._video_sampler_iterator:
            self._video_sampler_iterator = iter(MultiProcessSampler(self._video_sampler))

        if self._current_video:
            video, video_index = self._current_video
        else:
            video_index = next(self._video_sampler_iterator)

            video_frame_paths = [*self.root.joinpath(self._video_paths[video_index]).glob("*_y.png")]

            video = FrameVideo.from_frame_paths(video_frame_paths)

            self._current_video = video, video_index

        clip_info = self.clip_sampler(self._next_clip_start_sec, video.duration, {})

        if clip_info.aug_index == 0:
            self._current_video_clip = video.get_clip(
                clip_info.clip_start_sec,
                clip_info.clip_end_sec,
                self.frame_filter,
            )

        if clip_info.is_last_clip:
            self._current_video = None

            self._next_clip_start_sec = 0.0
        else:
            self._next_clip_start_sec = clip_info.clip_end_sec

        sample = {
            "aug_index": clip_info.aug_index,
            "clip_index": clip_info.clip_index,
            "label": None,
            "video": self._current_video_clip["video"],
            "video_index": video_index,
            "video_label": None,
            "video_name": str(self._video_paths[video_index]),
        }

        return sample

    def __len__(self) -> int:
        return len(self._video_paths)

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
