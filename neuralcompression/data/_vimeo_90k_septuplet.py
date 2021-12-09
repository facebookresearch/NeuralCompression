# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.transforms import ToTensor


class Vimeo90kSeptuplet(Dataset):
    """
    Loads images or videos from the Vimeo-90k septuplet dataset [1]. The dataset
    consists of a set of septuplet directories, where each directory contains
    seven consecutive frames of a video from vimeo.com. Each frame is stored as
    a PNG file. This class can be configured to return videos (consecutive
    frames in a septuplet) or individual images every time an item is accessed.

    Xue, Tianfan, et al. "Video enhancement with task-oriented flow."
    International Journal of Computer Vision 127.8 (2019): 1106-1125.

    Note:
        Following the conventions of ``torchvision``, in video mode this
        dataset will have a default transform of
        ``torchvision.transforms.ToTensor``, while in image mode no default
        transform is provided.

    Args:
        root: Path to the Vimeo-90k root directory (i.e. the
            directory containing the dataset's README).
        as_video: Determines whether the dataset should return
            individual images (``as_video=False``) or
            multiple consecutive frames  (``as_video=True``)
            at a time.
        frames_per_group: The number of frames to include from
            each septuplet. Specifically, the first ``frames_per_group``
            frames from each septuplet are included in
            the dataset. Must be between 1 and 7.
        split: Specifies which dataset parition should be used. Valid values
            are ``"train"`` or ``"test"``. Exactly one of ``split`` or
            'folder_list' must be specified.
        folder_list: A list of paths to septuplets to include in the dataset
            split. Each septuplet path must be a directory containing the files
            ``im1.png``, ..., ``im7.png``. Exactly one of ``split`` or
            ``folder_list`` must be specified.
        pil_transform: Callable object for applying transforms to
            the PIL images prior to image concatenation. If using, be sure to
            have the final operation convert the PIL image to a tensor.
            Following ``torchvision``'s dataset conventions, the default
            transform is ``torchvision.transforms.ToTensor`` when
            in video mode (i.e. when ``as_video=True``), while no
            default transform is applied in image mode.
        tensor_transform: Callable object for applying PyTorch
            transforms after data conversion and septuplet concatenation.
    """

    def __init__(
        self,
        root: Union[str, os.PathLike],
        as_video: bool = False,
        frames_per_group: int = 7,
        split: Optional[str] = None,
        folder_list: Optional[Sequence[str]] = None,
        pil_transform: Optional[Callable[[Any], Tensor]] = None,
        tensor_transform: Optional[Callable[[Tensor], Tensor]] = None,
    ):
        self.root = Path(root)
        self.as_video = as_video

        if frames_per_group not in list(range(1, 8)):
            raise ValueError(
                "frames_per_group must be an integer between 1 and 7 (inclusive), "
                f"not '{frames_per_group}'"
            )

        self.frames_per_group = frames_per_group
        self.pil_transform = (
            ToTensor() if pil_transform is None and as_video else pil_transform
        )
        self.tensor_transform = tensor_transform

        if (split is None) == (folder_list is None):
            raise ValueError("Exactly one of 'split', 'folder_list' must be specified.")

        if folder_list is not None:
            self.folder_list = folder_list
        else:
            if split not in ["train", "test"]:
                raise ValueError(
                    "split must take on values of either 'train' or 'test', "
                    f"not {split}"
                )

            with open(self.root / f"sep_{split}list.txt", "r") as file:
                self.folder_list = [
                    "sequences/" + fname.strip() for fname in file.readlines()
                ]

    def __len__(self) -> int:
        if self.as_video:
            return len(self.folder_list)
        else:
            return len(self.folder_list) * self.frames_per_group

    def load_image(self, septuplet: Path, frame_number: int) -> Tensor:
        img_path = default_loader(str(septuplet / f"im{frame_number}.png"))
        img = self.pil_transform(img_path)
        if not isinstance(img, Tensor):
            raise RuntimeError(
                "Tensor not returned from pil_transform. "
                "Did you forget to add ToTensor() to your transform?"
            )
        return img

    def __getitem__(self, idx: int) -> Tensor:
        if self.as_video:
            folder = self.root / self.folder_list[idx]
            images = []
            for im_num in range(1, self.frames_per_group + 1):
                images.append(self.load_image(folder, im_num))

            item: Tensor = torch.stack(images)

        else:
            folder_idx = idx // self.frames_per_group
            frame_idx = idx % self.frames_per_group + 1

            folder = self.root / self.folder_list[folder_idx]
            item = self.load_image(folder, frame_idx)

        if self.tensor_transform is not None:
            item = self.tensor_transform(item)

        return item
