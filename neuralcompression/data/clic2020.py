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


class CLIC2020(torch.utils.data.Dataset):
    resources = {
        "training": {
            "digest": "3f196ab93fc77d97bc99661c1cb1cfb983f17770",
            "endpoint": "https://data.vision.ee.ethz.ch/cvl/clic/professional_train_2020.zip",
            "source": "train",
        },
        "validation": {
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
            root: typing.Union[str, pathlib.Path],
            partition: str = "training",
            check_hash: bool = True,
            download: bool = False,
            transform: typing.Optional[typing.Callable[[typing.Any], torch.Tensor]] = None,
    ):
        if not pathlib.Path(root).exists():
            raise ValueError

        self.root = pathlib.Path(root, "clic2020")

        if partition not in ("training", "validation", "test"):
            raise ValueError

        self.partition = partition

        self.check_hash = check_hash

        self.transform = transform

        self.resource = self.resources[self.partition]

        self.source = self.root.joinpath(self.resource["source"])

        self.destination = self.root.joinpath(self.partition)

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
            if self.check_hash:
                self._check_integrity(archive, self.resource["digest"])

            if self.partition == "test":
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
