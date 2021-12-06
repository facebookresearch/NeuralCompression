# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional, Sequence, Union

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms

from neuralcompression.data import Vimeo90kSeptuplet


class Vimeo90kSeptupletLightning(LightningDataModule):
    """
    PyTorch Lightning data module version of ``Vimeo90kSeptuplet``.

    Note:
        The ``Vimeo90kSeptuplet`` dataloader by default will return images in
        the range ``[0, 1)`` without needing to add a normalization transform.

    Args:
        data_dir: root directory of Vimeo dataset.
        frames_per_group: Number of frames to load for the batch.
        train_batch_size: The batch size to use during training.
        val_batch_size: The batch size to use during validation.
        image_size: The size of the crop to take from the original images.
        num_workers: The number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: Whether prepared items should be loaded into pinned memory
            or not. This improves performance on GPUs.
    """

    def __init__(
        self,
        data_dir: str,
        frames_per_group: int = 2,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        image_size: Union[int, Sequence[int]] = 256,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        if frames_per_group not in list(range(1, 8)):
            raise ValueError(
                f"Received frames_per_group of {frames_per_group}, "
                "must be int in [0, ..., 7]."
            )
        self.data_dir = data_dir
        self.frames_per_group = frames_per_group
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        train_transforms = transforms.Compose(
            [transforms.CenterCrop(self.image_size), transforms.ToTensor()]
        )

        val_transforms = transforms.Compose(
            [transforms.CenterCrop(self.image_size), transforms.ToTensor()]
        )

        self.train_dataset = Vimeo90kSeptuplet(
            root=self.data_dir,
            frames_per_group=self.frames_per_group,
            pil_transform=train_transforms,
            as_video=True,
            split="train",
        )

        self.val_dataset = Vimeo90kSeptuplet(
            self.data_dir,
            frames_per_group=self.frames_per_group,
            pil_transform=val_transforms,
            as_video=True,
            split="test",
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
