# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import List, Optional, Union

import pytorchvideo.data
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from datamodules.video_data_api import KineticsDataset
from pytorch_lightning import LightningDataModule
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RemoveKey,
    UniformTemporalSubsample,
)
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    RandomChoice,
    RandomCrop,
    RandomHorizontalFlip,
    RandomResizedCrop,
)


class KineticsDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        train_batch_size: int = 4,
        val_batch_size: int = 4,
        clip_duration: int = 2,
        video_num_subsampled: Optional[Union[int, List[int]]] = None,
        normalize: bool = False,
        train_clip_sampler: str = "uniform",
        num_workers: int = 2,
        pin_memory: bool = True,
    ) -> None:
        """Kinetics datamodule.

        Args:
            data_dir: directory containing the kinetics dataset.
            train_batch_size, val_batch_size: batch size for train/val set.
            clip_duration: duration of sampled clip for each video.
            video_num_subsampled: number of frames to subsample from each video using
                `UniformTemporalSubsample`. If None, no subsampling is applied.
            normalize: images are always scaled (/255) and optionally normalized to
            (approximately) zero mean and unit std.
            train_clip_sampler: choose "random" for typical classification objective.
                Choose "uniform" to sample all clips of a given duration from video,
                useful for SSL objective. See `pytorchvideo.data.make_clip_sampler` for
                more details.
            num_workers: number of parallel processes fetching data.
            pin_memory: if `True`, the data loader will copy Tensors
                into device/CUDA pinned memory before returning them.
                See pytorch DataLoader for more details.
        """

        super().__init__()
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.clip_duration = clip_duration
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_clip_sampler = train_clip_sampler
        self.video_num_subsampled = video_num_subsampled
        if video_num_subsampled is not None:
            self.max_len = (
                video_num_subsampled
                if isinstance(video_num_subsampled, int)
                else max(video_num_subsampled)
            )
        # The mean and std are the default in pytorchvideo.transforms.create_video_transform
        # Note data is first /255 and then pytorchvideo.transforms.Normalize-d
        # Omnivision uses [123.675, 116.28, 103.53]; [58.395, 57.12, 57.375]
        self.normalize = normalize
        if normalize:
            self.video_means = (0.45, 0.45, 0.45)
            self.video_stds = (0.225, 0.225, 0.225)
        self.video_crop_size = 256
        self.video_min_short_side_scale = 256
        self.video_max_short_side_scale = 320

    def _custom_collate_val(self, batch) -> KineticsDataset:
        """
        batch: list of length batch_size
        """
        # transpose: input is [C, T, H, W] -> [T, C, H, W]
        for b in batch:
            b["video"] = b["video"].permute(1, 0, 2, 3).contiguous()
        batch = torch.utils.data.default_collate(batch)
        return KineticsDataset(video_tensor=batch["video"])

    def _custom_collate_train(self, batch, pad: bool = False) -> KineticsDataset:
        """
        batch: list of length batch_size
        """
        sequences = []
        # transpose: input is [C, T, H, W] -> [T, C, H, W]
        for b in batch:
            b["video"] = b["video"].permute(1, 0, 2, 3).contiguous()
            if pad:
                sequences.append(b["video"])

        if pad:
            padded_batch = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
            # Hack this for now to avoid rewriting more code
            for b, pb in zip(batch, padded_batch):
                b["video"] = pb

        batch = torch.utils.data.default_collate(batch)
        return KineticsDataset(video_tensor=batch["video"])

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = pytorchvideo.data.Kinetics(
            data_path=os.path.join(self.data_dir, "train"),
            clip_sampler=pytorchvideo.data.make_clip_sampler(
                self.train_clip_sampler, self.clip_duration
            ),
            decode_audio=False,
            transform=self._video_transform(mode="train"),
        )
        self.val_dataset = pytorchvideo.data.Kinetics(
            data_path=os.path.join(self.data_dir, "val"),
            clip_sampler=pytorchvideo.data.make_clip_sampler(
                "uniform", self.clip_duration
            ),
            decode_audio=False,
            transform=self._video_transform(mode="val"),
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create the Kinetics train partition from the list of video labels
        in {self.data_dir}/train. Add transform that subsamples and normalizes the
        video before applying the scale, crop and flip augmentations
        """
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._custom_collate_train,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create the Kinetics validation partition from the list of video labels
        in {self.data_dir}/val
        """
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._custom_collate_val,
        )

    def test_dataloader(self, test_batch_size: int = 1) -> torch.utils.data.DataLoader:
        """
        Create the Kinetics train partition from the list of video labels
        in {self.data_dir}/test
        """
        transforms = self._video_transform(mode="test")

        test_dataset = pytorchvideo.data.Kinetics(
            data_path=os.path.join(self.data_dir, "test"),
            clip_sampler=pytorchvideo.data.make_clip_sampler(
                self.train_clip_sampler, self.clip_duration
            ),
            decode_audio=False,
            transform=transforms,
        )
        return torch.utils.data.DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._custom_collate_val,
        )

    def _video_transform(self, mode: str) -> torchvision.transforms.Compose:
        """
        The same TorchVision transforms are applied across all rames in the clip (e.g.
        position of the random crop is constant across clip frames). Transforms we
        apply include:

            Temporal Subsampling: unofromly sunsample `video_num_subsampled` frames
            Scaling: scale rgb [0, 255] to [0, 1]
            Normalization: if `notmalize`, normalize with imagenet mean and scale
            Augmentations:
                - In 'train' random augmentations -- (resized) crop, horizontal flip)
                - In 'val' -- center crop only
                - In 'test' -- no augmentations
        """
        if self.video_num_subsampled is None:
            # No subsampling for val and test modes
            temporal_subsampling = []
        elif isinstance(self.video_num_subsampled, int):
            temporal_subsampling = [
                UniformTemporalSubsample(num_samples=self.video_num_subsampled)
            ]
        elif mode == "train":
            temporal_subsampling = [
                RandomChoice(
                    [
                        UniformTemporalSubsample(num_samples=num_samples)
                        for num_samples in self.video_num_subsampled
                    ]
                )
            ]
        else:
            temporal_subsampling = [UniformTemporalSubsample(num_samples=self.max_len)]

        scaling = [Lambda(lambda x: x / 255.0)]
        normalization = (
            [Normalize(self.video_means, self.video_stds)] if self.normalize else []
        )
        if mode == "train":
            augmentations = [
                RandomChoice(
                    [
                        RandomCrop(
                            size=self.video_crop_size,
                            pad_if_needed=True,
                            padding_mode="edge",
                        ),
                        RandomResizedCrop(size=self.video_crop_size, scale=(0.6, 1)),
                    ]
                ),
                RandomHorizontalFlip(p=0.5),
            ]
        elif mode == "val":
            augmentations = [CenterCrop(size=self.video_crop_size)]
        else:
            augmentations = []  # no augmentations for val and test modes

        video_transforms = ApplyTransformToKey(
            key="video",
            transform=Compose(
                temporal_subsampling + scaling + normalization + augmentations
            ),
        )
        return Compose([video_transforms, RemoveKey("audio")])
