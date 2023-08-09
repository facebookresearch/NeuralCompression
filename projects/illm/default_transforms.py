# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torchvision.transforms import (
    CenterCrop,
    Compose,
    RandomChoice,
    RandomCrop,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)


def default_train_transform(image_size: int) -> Compose:
    choice_transform = RandomChoice(
        [
            RandomCrop(size=image_size, pad_if_needed=True, padding_mode="reflect"),
            RandomResizedCrop(size=image_size),
        ]
    )
    return Compose(
        [
            choice_transform,
            RandomHorizontalFlip(),
            ToTensor(),
        ]
    )


def default_val_transform(image_size: int) -> Compose:
    return Compose(
        [
            Resize(image_size),
            CenterCrop(image_size),
            ToTensor(),
        ]
    )
