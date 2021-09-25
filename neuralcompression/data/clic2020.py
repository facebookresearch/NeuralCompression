import os
import pathlib
import shutil
import typing

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
            "filename": "train.zip",
            "md5": "a6845cac88c3dd882246575f7a2fc5f9",
            "url": "https://data.vision.ee.ethz.ch/cvl/clic/professional_train_2020.zip",
        },
        "val": {
            "filename": "val.zip",
            "md5": "7111ee240435911db04dbc5f40d50272",
            "url": "https://data.vision.ee.ethz.ch/cvl/clic/professional_valid_2020.zip",
        },
        "test": {
            "filename": "test.zip",
            "md5": "4476b708ea4c492505dd70061bebe202",
            "url": "https://storage.googleapis.com/clic2021_public/professional_test_2021.zip",
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

        self.split = torchvision.datasets.utils.verify_str_arg(split, "split", ("train", "val", "test"))

        self.transform = transform

        self.resource = self.resources[self.split]

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
        kwargs = {
            "download_root": self.root,
            "extract_root": self.root,
            **self.resource
        }

        if self.split == "test":
            kwargs["extract_root"] = self.root.joinpath("test")

        torchvision.datasets.utils.download_and_extract_archive(**kwargs)

        if self.split == "val":
            os.rename(self.root.joinpath("valid"), self.destination)

        if self.split in {"train", "val"}:
            shutil.rmtree(self.root.joinpath("__MACOSX"))
        else:
            shutil.rmtree(self.destination.joinpath("__MACOSX"))
