# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.utils.data
import torchvision
from datamodules.video_data_api import VideoDataset
from pytorch_lightning import LightningDataModule
from pytorchvideo.transforms import Normalize
from torchvision.transforms import Compose, ToTensor


class UVGDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        num_frames: Optional[int] = None,
        normalize: bool = False,
        num_workers: int = 2,
        pin_memory: bool = True,
    ) -> None:
        """UVG evaluation dataset datamodule.

        UVG consists of 6 5-second long high-resolution videos (w,h=1920x1080) at 120fps
        and 1 video (SHakeNDry) which is 2.5 seconds. Total number of frames is in the
        dataset is 3900 (=6*600 + 300).
        This is an evaluation dataset only. As such the module always loads videos in a
        bit of a strange way, specifically:
            - always does batch size 1
            - the frames per video is determined by <num_frames>, which is required to
              divide 300 in order to avoid mixing multiple videos in the same sample.
        This is essentially an image dataloader that loads images in a video-like format.

        Args:
            data_dir: directory to UVG dataset stored as images.
            num_frames: number of frames from a single video to load. If None, returns
                300 frames per video, i.e. sample will be of size [1, 300, 3, 1080, 1920]
                If specified, the sample is of size [1, num_frames, 1080, 1920], and
                num_frames must divide 300 to avoid mixing different videos in a sample.
            normalize: images are scaled (/255) and can be optionally normalized to
                (approximately) zero mean and unit std.
            num_workers: number of parallel processes fetching data.
            pin_memory: if `True`, the data loader will copy Tensors
                into device/CUDA pinned memory before returning them.
                See pytorch DataLoader for more details.
        """

        super().__init__()
        self._frames_per_video = 300
        self._total_num_videos = 13  # 6 at 600 and 1 at 300 frames = 13 at 300
        self.data_dir = data_dir
        self._batch_size = (
            num_frames if num_frames is not None else self._frames_per_video
        )
        assert (
            self._frames_per_video % self._batch_size == 0
        ), "Invalid video_num_subsampled, must divide 300"
        self._total_batches = (
            self._frames_per_video * self._total_num_videos // self._batch_size
        )
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # The mean and std are the default in pytorchvideo.transforms.create_video_transform
        # Note data is first /255 and then pytorchvideo.transforms.Normalize-d
        # Omnivision uses [123.675, 116.28, 103.53]; [58.395, 57.12, 57.375]
        self.normalize = normalize
        if normalize:
            self.video_means = (0.45, 0.45, 0.45)
            self.video_stds = (0.225, 0.225, 0.225)

    def _custom_collate(self, batch) -> VideoDataset:
        video, labels = torch.utils.data.default_collate(batch)
        # transform to VideoData format
        assert (labels == labels[0]).all(), "Error: more than one video in a sample"
        batch = {"video": video, "label": labels}
        batch = torch.utils.data.default_collate([batch])
        return VideoDataset(video_tensor=batch["video"])

    def _transforms(self) -> torchvision.transforms.Compose:
        """
        covnert image to tensor and optionally normalize with Imagenet means and stds
        """
        totensor = [ToTensor()]
        normalize = (
            [Normalize(self.video_means, self.video_stds)] if self.normalize else []
        )
        return Compose(totensor + normalize)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """
        This is only an evaluation dataset, so there should be only test_dataloader
        """
        dataset = torchvision.datasets.ImageFolder(
            root=self.data_dir, transform=self._transforms()
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self._batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._custom_collate,
        )
