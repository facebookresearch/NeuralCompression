# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
from typing import Callable, List, Optional, Union

from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

#  very large or corrupted images in train
_INVALID_IMAGE_NAMES = {
    "0a866d6fd1ea9317",
    "0df79dea97961a10",
    "0efa17ee62d7caf9",
    "14098132e2032aaf",
    "193edd6c8ef39251",
    "1b564e7e54361fc2",
    "24b77bff6bfd736c",
    "39a876a07f18f837",
    "3e260ed1bfd6509e",
    "42a6df3ac3bf2d33",
    "4ce068dcc2ae3cc9",
    "4ddbede0f1bdc602",
    "50eb15613b100675",
    "5ef177f1391765d3",
    "61efc69916e8fa1e",
    "62a1bbca8a844225",
    "62be906e5077bffa",
    "646605f070b9883f",
    "67b1dfe383f92dcf",
    "6b1efb7bf49e27be",
    "73497cedb79d23a4",
    "73f886b2ab2f5bd9",
    "79bcae5f044a69fb",
    "80d19062a39df158",
    "83c0499d3e8fdb06",
    "8604fd077594d226",
    "88161ed23f3379b4",
    "895ef7a64e2858c1",
    "91d66b5ddcb4f5a3",
    "93fd5f0899065d78",
    "965b4dccff47fa5d",
    "add170ca187afcd7",
    "b6474a1a69bbbe49",
    "b840cb6cca32d756",
    "bcd2dfa3750ea055",
    "bebf5b2470a3955c",
    "c18b0a385e0d8257",
    "c5ad20115f79cc66",
    "d1fdbbb705bb6016",
    "d66352405f0385e3",
    "da1267b1196c8e79",
    "ea15e23587edc562",
    "eb8fab01d113dc3b",
    "ec1a5e86a7d570c8",
    "ec40941f60fbc594",
    "ecd25d7e90027c89",
    "fb6574ec2e4e4bb8",
    "33bbb7e2c767c773",
    "9f860289e93d11fc",
    "c977b4fd166e908b",
    "c2ffdc701a70e9a1",
    "0248001275c71dc8",
    "0364f5c6f964e1bc",
    "0b76aafb249fc5b7",
    "0c616f7947875acc",
    "16f199c310a1b7ac",
    "235f7b87aa67461e",
    "23e5990f0ed037a7",
    "24d76ef27ba41f38",
    "2e58f4a0d31e9be1",
    "2e7f9bc07e7bd4d8",
    "2ff06b2f1ba0cc51",
    "31968a4b02d9981b",
    "36a83a554ef21da2",
    "39af00cf7013519c",
    "3dfceab39d674168",
    "439e232d7531027e",
    "48212be4d45c5c1a",
    "59847832eb83aaa1",
    "5cbd57fd38a45852",
    "61db1e0c8174b3c3",
    "659c55b26b92c5ab",
    "6a5bdb01a0f99891",
    "6bba1f25a1ee68e8",
    "723a41acf8804460",
    "752cf54040dc80d7",
    "7996ed8f1f35ce83",
    "8075443714f28729",
    "8b1f16626a5bbc82",
    "8d7efba30eeca417",
    "97e9a3774468eb35",
    "a7adff7aede82670",
    "adcd441c940a8832",
    "ae811292e322abe1",
    "b146377ed3af77ec",
    "b8543f9c563bad6a",
    "b9511a25bf6aa6d9",
    "b9e6642fc707d78f",
    "bf8844bdf5fcded4",
    "c100f8bda2d56a6d",
    "c2ffdc701a70e9a1",
    "c57eb29e74376b15",
    "c977b4fd166e908b",
    "cbcdc984e6f0d41d",
    "cf81b04e3baf23d8",
    "d45776312fb97ff6",
    "d5af6277007b6001",
    "daa2fa027bd35189",
    "de2ea1f167b03dab",
    "dfc42b77003e55d3",
    "e752e43deb95c51c",
    "f0508277b718aa55",
}


def _load_indices(indices_path: Path, split: str) -> List[str]:
    with open(indices_path, "r") as f:
        indices = sorted(f.readlines())

    image_names = [os.path.splitext(el.rstrip())[0] for el in indices]
    if split == "train":
        image_names = [el for el in image_names if el not in _INVALID_IMAGE_NAMES]

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
        else:
            self.image_list_file = Path(image_list_file)

        self.indices = _load_indices(self.image_list_file, split=split)

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
