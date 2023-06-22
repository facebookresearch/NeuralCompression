# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Callable, Optional, Union

import torchvision.datasets.folder
import torchvision.datasets.utils
from PIL.Image import Image
from torch import Tensor
from torch.utils.data import Dataset


class DIV2KDataset(Dataset):
    """
    Dataloader for DIV2K Dataset.

    The DIV2K dataset was released for the 2017 NTIRE challenge on single-image
    super resolution. The dataset is documented in the following paper:

    NTIRE 2017 challenge on single image super-resolution: Dataset and study
    E Agustsson, R Timofte

    Args:
        root: Root directory of dataset.
        split: Split of the dataset (currently only 'val' is accepted).
        transform: Torchvision PIL transform for converting the images.
    """

    split_map = {
        "val": "DIV2K_valid_HR",
    }

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "val",
        transform: Optional[Callable[[Image], Tensor]] = None,
    ):
        self.root: Path = Path(root)

        if split not in ("val",):
            raise ValueError("Invalid split.")

        self.split = self.split_map[split]
        self.transform = transform

        self.paths = sorted([*self.root.joinpath(self.split).glob("*.png")])

    def __getitem__(self, index: int) -> Image:
        path = self.paths[index]

        image = torchvision.datasets.folder.default_loader(path)

        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self) -> int:
        return len(self.paths)
