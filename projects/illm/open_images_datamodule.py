# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path
from typing import Callable, List, Optional

from default_transforms import default_train_transform, default_val_transform
from lightning.pytorch import LightningDataModule
from PIL.Image import Image
from torch import Tensor
from torch.utils.data import DataLoader

from neuralcompression.data import OpenImagesV6


class OpenImagesDataModule(LightningDataModule):
    def __init__(
        self,
        open_images_root,
        batch_size: int,
        workers: int = 0,
        image_size: Optional[List[int]] = None,
        train_transform: Optional[Callable[[Image], Tensor]] = None,
        val_transform: Optional[Callable[[Image], Tensor]] = None,
        full_size: bool = False,
        use_async: bool = False,
        pin_memory: bool = True,
        val_batch_size: Optional[int] = None,
    ):
        super().__init__()
        self.open_images_root = Path(open_images_root)
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.workers = workers
        self.full_size = full_size
        self.prepare_data_per_node = True
        self.use_async = use_async
        self.pin_memory = pin_memory
        self.logger = logging.getLogger(self.__class__.__name__)
        if train_transform is None:
            if image_size is None:
                raise ValueError("Must pass image_size if no train_transform.")
            self.train_transform = default_train_transform(image_size)
        else:
            self.train_transform = train_transform
        if val_transform is None:
            if image_size is None:
                raise ValueError("Must pass image_size if no val_transform.")
            self.val_transform = default_val_transform(image_size)
        else:
            self.val_transform = val_transform

    def setup(self, stage: Optional[str] = None):
        self.logger.info(f"OpenImages_V6 root: {self.open_images_root}")
        self.train_dataset = OpenImagesV6(
            root=str(self.open_images_root),
            split="train",
            transform=self.train_transform,
        )
        self.eval_dataset = OpenImagesV6(
            root=str(self.open_images_root),
            split="val",
            transform=self.val_transform,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=self.pin_memory,
            prefetch_factor=2,
        )

    def val_dataloader(self):
        if self.val_batch_size is not None:
            batch_size = self.val_batch_size
        else:
            batch_size = self.batch_size

        return DataLoader(
            self.eval_dataset,
            batch_size=batch_size,
            num_workers=self.workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return self.val_dataloader()
