# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import shutil
import urllib
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional, Union

from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from tqdm import tqdm

NUM_IMAGES = 24
SHA_HASH = "076e2a5fd4515b5fda677f15fe63dd112d266bc5"
DEFAULT_URL = "http://r0k.us/graphics/kodak/kodak/"


class Kodak(Dataset):
    """
    Data loader for the Kodak image data set.

    Args:
        root: base directory for data set.
        check_hash: if ``True``, checks the sha1 hash of root at
            initialization time to make sure files were downloaded correctly.
        download: if ``True``, downloads the dataset from the
            internet and puts it in root directory.
        force_download: if ``True`` and ``download=True``, will download
            the dataset even if the root directory already exists.
            If ``False``, an error will be thrown if ``download=True``
            but the root directory already exists.
        kodak_url: URL for downloading public images.
            Defaults to http://r0k.us/graphics/kodak/kodak/.
        transform: callable object for transforming the
            loaded images.
    """

    def __init__(
        self,
        root: Union[str, Path],
        check_hash: bool = True,
        download: bool = False,
        force_download: bool = False,
        kodak_url: str = DEFAULT_URL,
        transform: Optional[Callable[[Any], Tensor]] = None,
    ):
        self.root = Path(root)
        self.im_list = []
        self.transform = transform

        self.im_list = [
            self.root / "kodim{:02}.png".format(im_num)
            for im_num in range(1, NUM_IMAGES + 1)
        ]

        if download:
            self._download(kodak_url, NUM_IMAGES, force=force_download)
        elif not self.root.exists():
            raise FileNotFoundError(f"Directory {self.root} does not exist.")

        if check_hash:
            self._check_integrity(SHA_HASH)

    def _check_integrity(self, sha_hash):
        sha1 = hashlib.sha1()
        for img_name in self.im_list:
            with open(img_name, "rb") as f:
                data = f.read()
                sha1.update(data)

        if sha1.hexdigest() != sha_hash:
            raise RuntimeError("Hash does not match. Kodak data may be corrupt.")

    def _download(self, kodak_url, num_imgs, force=False):
        if self.root.exists() and not force:
            raise RuntimeError(
                "Dataset directory already exists - to proceed"
                " with dataset download anyways pass force_download=True"
            )

        self.root.mkdir()

        for im_num in tqdm(range(1, num_imgs + 1), desc="Downloading Kodak images"):
            img_name = "kodim{:02}.png".format(im_num)
            local_file = self.root / img_name
            url = kodak_url + img_name
            with urllib.request.urlopen(url) as response, open(
                local_file, "wb"
            ) as out_file:
                shutil.copyfileobj(response, out_file)

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, idx):
        path = self.im_list[idx]
        image = default_loader(str(path))

        if self.transform is not None:
            image = self.transform(image)

        return image
