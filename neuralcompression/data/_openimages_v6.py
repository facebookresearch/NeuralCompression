# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
from typing import Callable, List, Optional, Union

from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


def _load_indices(indices_path: Path) -> List[str]:
    with open(indices_path, "r") as f:
        indices = sorted(f.readlines())
    image_names = [os.path.splitext(el.rstrip())[0] for el in indices]

    return image_names


class OpenImagesV6(Dataset):
    """
    Open Images V6 data loader

    This is a minimalist dataloader for Open Images V6. Since Open Images puts
    all data into a single folder, it is expected that the user has already
    created a text file with all the images of the split prior to instantiating
    this class (otherwise an ``os.walk`` takes forever). The train file should
    be called 'list_train_files.txt' by default, for example.

    If the user gives a different name to the image list file, the new filename
    can be specified in the ``__init__`` function.

    Args:
        root: Root direcotry for Open Images V6.
        split: Which split of the dataset (one of 'train', 'val', or 'test').
            This is used to indicate the subfolder of ``root`` where the images
            are stored, except for the case of 'val', which is renamed to
            'validation'
        image_list_file: The file containing all image names under the split.
            By default, it assumes the file has the name
            'list-{split}-files.txt'.
        transform: A torchvision PIL transform.
    """

    split_map = {"train": "train", "val": "validation", "test": "test"}

    def __init__(
        self,
        root: Union[str, Path],
        split: str,
        image_list_file: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        if split not in ("train", "val", "test"):
            raise ValueError("Split must be in ('train', 'val', 'test')")

        self.root = Path(root)
        self.split = self.split_map[split]
        self.transform = transform

        if image_list_file is None:
            self.image_list_file = self.root / f"list-{self.split}-files.txt"

        self.indices = _load_indices(self.image_list_file)

    def _get_image_path(self, index: int) -> Path:
        fname = self.indices[index]
        return self.root / self.split / (fname + ".jpg")

    def __getitem__(self, index: int):
        image_full_path = self._get_image_path(index)
        image = default_loader(image_full_path)

        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self) -> int:
        return len(self.indices)
