"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import sleep
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from urllib.request import urlopen, urlretrieve
from zipfile import ZipFile

from pytorchvideo.data.clip_sampling import ClipSampler
from pytorchvideo.data.frame_video import FrameVideo
from pytorchvideo.data.utils import MultiProcessSampler
from torch import Tensor, clamp, linspace
from torch.utils.data import IterableDataset, RandomSampler, Sampler
from torchvision.datasets.utils import verify_str_arg
from tqdm import tqdm


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
        video_sampler:
        transform: A callable that transforms a clip, e.g,
            ``pytorchvideo.transforms.RandomResizedCrop``.
        multithreaded_io: If true, performs input/output operations across
            multiple threads.
        frames_per_clip:
    """
    _current_video: Optional[Tuple[FrameVideo, int]] = None
    _current_video_clip: Optional[Dict[str, Optional[Tensor]]]
    _destination_root = "https://storage.googleapis.com/clic2021_public/txt_files"
    _next_clip_start_sec: float = 0.0

    def __init__(
        self,
        root: Union[str, Path],
        clip_sampler: ClipSampler,
        split: str = "train",
        download: bool = False,
        transform: Optional[Callable[[Dict], Dict]] = None,
        video_sampler: Type[Sampler] = RandomSampler,
        multithreaded_io: bool = False,
        frames_per_clip: Optional[int] = None,
    ):
        self._root = Path(root)

        self._clip_sampler = clip_sampler

        self._split = verify_str_arg(split, "split", ("train", "val", "test"))

        self._transform = transform

        if download:
            self.download()

        self._video_paths = [Path(path.name) for path in self._root.glob("*")]

        self._video_sampler = video_sampler(self._video_paths)

        self._video_sampler_iterator = iter(MultiProcessSampler(self._video_sampler))

        if frames_per_clip:
            self._frames_per_clip = frames_per_clip

            self._frame_filter = self._sample_frames

        self._next_clip_start_sec = 0.0

    def __getitem__(self, index: int) -> Dict:
        raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self) -> Dict:
        if not self._video_sampler_iterator:
            self._video_sampler_iterator = iter(
                MultiProcessSampler(self._video_sampler)
            )

        if self._current_video:
            video, index = self._current_video
        else:
            index = next(self._video_sampler_iterator)

            video_frame_paths = [
                *self._root.joinpath(self._video_paths[index]).glob("*_y.png")
            ]

            video = FrameVideo.from_frame_paths(video_frame_paths)

            self._current_video = video, index

        clip_info = self._clip_sampler(self._next_clip_start_sec, video.duration, {})

        if clip_info.aug_index == 0:
            self._current_video_clip = video.get_clip(
                clip_info.clip_start_sec,
                clip_info.clip_end_sec,
                self._frame_filter,
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
            "video_index": index,
            "video_label": None,
            "video_name": str(self._video_paths[index]),
        }

        if self._transform:
            sample = self._transform(sample)

        return sample

    def __len__(self) -> int:
        return len(self._video_paths)

    def download(self):
        self._root.mkdir(exist_ok=True, parents=True)

        with urlopen(f"{self._destination_root}/video_urls.txt") as file:
            endpoints = file.read().decode("utf-8").splitlines()

        def f(endpoint: str):
            sleep(0.001)

            path, _ = urlretrieve(endpoint)

            with ZipFile(path, "r") as archive:
                archive.extractall(self._root)

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

    def _sample_frames(self, frames: List[int]) -> List[int]:
        n = len(frames)

        indicies = clamp(linspace(0, n - 1, self._frames_per_clip), 0, n - 1).long()

        return [frames[index] for index in indicies]
