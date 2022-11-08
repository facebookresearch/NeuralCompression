"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Prepare Cifar10 data.

Adapted from `improved-diffusion.datasets.cifar10`
(https://github.com/openai/improved-diffusion)
"""
import argparse
import logging
import tempfile
from pathlib import Path

import torchvision
from tqdm.auto import tqdm

CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

SIZE = {
    "train": 50000,
    "valid": 10000,
}


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("cifar10")
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out_dir", help="Path to output directory.", type=str)
    parser.add_argument(
        "-w", "--overwrite", help="Overwrite existing files.", action="store_true"
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = Path(__file__).parent / "cifar10"
    else:
        out_dir = Path(out_dir)

    for split in ["train", "valid"]:
        split_dir = out_dir / f"{split}_32x32"
        if split_dir.is_dir() and not args.overwrite:
            logger.info("skipping split %s since %s already exists.", split, split_dir)
            continue

        logger.info("downloading...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = torchvision.datasets.CIFAR10(
                root=tmp_dir, train=split == "train", download=True
            )
        assert len(dataset) == SIZE[split]

        logger.info("dumping images...")
        split_dir.mkdir(parents=True, exist_ok=True)
        for i in tqdm(range(len(dataset))):
            image, label = dataset[i]
            filename = split_dir / f"{CLASSES[label]}_{i:05d}.png"
            image.save(filename)


if __name__ == "__main__":
    main()
