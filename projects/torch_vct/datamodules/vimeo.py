# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Sequence, Union

import torchvision
from datamodules.video_data_api import VimeoDataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, default_collate
from torchvision.transforms import (
    CenterCrop,
    Compose,
    RandomChoice,
    RandomCrop,
    RandomHorizontalFlip,
    RandomResizedCrop,
)

from neuralcompression.data import Vimeo90kSeptuplet


class VimeoDataModule(LightningDataModule):
    """
    PyTorch Lightning data module version of ``Vimeo90kSeptuplet``.

    Args:
        data_dir: root directory of Vimeo dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This improves performance on GPUs.
    """

    def __init__(
        self,
        data_dir: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        video_crop_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 2,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.video_crop_size = video_crop_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def _video_transform(self, mode: str) -> torchvision.transforms.Compose:
        scaling = []
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
        else:
            augmentations = [CenterCrop(size=self.video_crop_size)]

        return Compose(scaling + augmentations)

    def _custom_collate(self, batch) -> VimeoDataset:
        batch = default_collate(batch)
        return VimeoDataset(video_tensor=batch)

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = Vimeo90kSeptuplet(
            root=self.data_dir,
            as_video=True,
            tensor_transform=self._video_transform(mode="train"),
            split="train",
        )

        self.val_dataset = Vimeo90kSeptuplet(
            root=self.data_dir,
            as_video=True,
            tensor_transform=self._video_transform(mode="test"),
            split="test",
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            collate_fn=self._custom_collate,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            collate_fn=self._custom_collate,
        )
