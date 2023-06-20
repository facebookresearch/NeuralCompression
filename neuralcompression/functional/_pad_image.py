# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch.nn.functional as F
from torch import Tensor


def pad_image_to_factor(
    image: Tensor, factor: int, mode: str = "reflect"
) -> Tuple[Tensor, Tuple[int, int]]:
    """
    Pads an image if it is not divisible by factor.

    For many neural autoencoders, performance suffers if the input image is not
    divisible by the downsampling factor. This utility function can be used to
    pad the input image with reflection padding to avoid such cases.

    Args:
        image: A 4-D PyTorch tensor with dimensions (B, C, H, W)
        factor: A factor by which the output image should be divisible.

    Returns:
        The image padded so that its dimensions are disible by factor, as well
        as the height and width.
    """
    # pad image if it's not divisible by downsamples
    _, _, height, width = image.shape
    pad_height = (factor - (height % factor)) % factor
    pad_width = (factor - (width % factor)) % factor
    if pad_height != 0 or pad_width != 0:
        image = F.pad(image, (0, pad_width, 0, pad_height), mode=mode)

    return image, (height, width)
