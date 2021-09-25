import hashlib
import os
import pathlib
import shutil
import typing
import urllib.request
import zipfile

import PIL.Image
import torch.utils.data
import torchvision.datasets.folder
import torchvision.datasets.utils


class CLIC2020(torch.utils.data.Dataset):
    """`Challenge on Learned Image Compression (CLIC) 2020 <http://compression.cc/tasks/>`_ Dataset.

    Args:
        root (str): Root directory where images are downloaded to.
            Expects the following folder structure if download=False:

            .. code::

                <root>
                    └── clic2020
                        ├── test
                        │   ├── *.png
                        ├── train
                        │   ├── *.png
                        └── val
                            └── *.png
        split (str): The dataset split to use. One of {``train``, ``val``, ``test``}.
            Defaults to ``train``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version.  E.g, ``transforms.RandomCrop``.
    """
    resources = {
        "train": {
            "digest": "3f196ab93fc77d97bc99661c1cb1cfb983f17770",
            "endpoint": "https://data.vision.ee.ethz.ch/cvl/clic/professional_train_2020.zip",
            "source": "train",
        },
        "val": {
            "digest": "196a602548afcc1949e5ec3a5dd0f6e713b34201",
            "endpoint": "https://data.vision.ee.ethz.ch/cvl/clic/professional_valid_2020.zip",
            "source": "valid",
        },
        "test": {
            "digest": "6dadb31eeae6cee8d2455a35818129eb68b0adbb",
            "endpoint": "https://storage.googleapis.com/clic2021_public/professional_test_2021.zip",
            "source": ".",
        }
    }

    def __init__(
            self,
            root: str,
            split: str = "train",
            download: bool = False,
            transform: typing.Optional[typing.Callable] = None,
    ):
        if not pathlib.Path(root).exists():
            raise ValueError

        self.root: pathlib.Path = pathlib.Path(root, "clic2020")

        self.split = torchvision.datasets.utils.verify_str_arg(
            split, "split", ("train", "val", "test")
        )

        self.transform = transform

        self.resource = self.resources[self.split]

        self.source = self.root.joinpath(self.resource["source"])

        self.destination = self.root.joinpath(self.split)

        if self.destination.exists() and not download:
            raise RuntimeError

        self.root.mkdir(exist_ok=True)

        if download:
            self.download()

        self.paths = [*self.destination.glob("*.png")]

    def __getitem__(self, index: int) -> PIL.Image.Image:
        path = self.paths[index]

        image = torchvision.datasets.folder.default_loader(path)

        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self) -> int:
        return len(self.paths)

    def download(self):
        path, _ = urllib.request.urlretrieve(self.resource["endpoint"])

        with zipfile.ZipFile(path, "r") as archive:
            self._check_integrity(archive, self.resource["digest"])

            if self.split == "test":
                archive.extractall(self.destination)

                shutil.rmtree(self.destination.joinpath("__MACOSX"))

                return

            archive.extractall(self.root)

            os.rename(self.source, self.destination)

            shutil.rmtree(self.root.joinpath("__MACOSX"))

    @staticmethod
    def _check_integrity(archive: zipfile.ZipFile, checksum: str):
        sha1 = hashlib.sha1()

        while True:
            chunk = archive.fp.read(16 * 1024)

            if not chunk:
                break

            sha1.update(chunk)

        if sha1.hexdigest() != checksum:
            raise RuntimeError
