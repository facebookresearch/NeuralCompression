# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import matplotlib.pyplot as plt
import torch
from torch import Tensor


def show_image(
    image: Tensor,
    mean: Tensor = torch.zeros(3),
    std: Tensor = torch.tensor([255] * 3),
) -> None:
    # image has to be [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip(image.cpu() * std.cpu() + mean.cpu(), 0, 255).int())
    plt.axis("off")
    return


def show_video(
    clip: Tensor,
    nrow: int = 1,
    mean: Tensor = torch.zeros(3),
    std: Tensor = torch.tensor([255] * 3),
) -> None:
    """
    clip: expected shape [T, H, W, C]
    """
    assert clip.shape[3] == 3
    num_frames = clip.shape[0]
    fps = math.ceil(num_frames / nrow)

    plt.figure(figsize=(fps * 2, nrow * 2))
    for i in range(num_frames):
        plt.subplot(nrow, fps, i + 1)
        show_image(clip[i, ...], mean=mean, std=std)
    return
