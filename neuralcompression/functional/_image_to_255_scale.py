# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch import Tensor


def image_to_255_scale(image: Tensor, dtype: Optional[torch.dtype] = None) -> Tensor:
    """
    Helper function for converting a floating point image to 255 scale.

    The input image is expected to be in the range [0.0, 1.0]. If it is outside
    this range, the function throws an error.

    Args:
        image: A 4-D PyTorch tensor.
        dtype: Output datatype. If not passed, the image is of the same dtype
            as the input.

    Returns:
        The image in [0, 255] scale.
    """
    if image.max() > 1.0:
        raise ValueError("Unexpected image max > 1.0")
    if image.min() < 0.0:
        raise ValueError("Unexpected image min < 0.0")

    image = torch.round(image * 255.0)

    if dtype is not None:
        image = image.to(dtype)

    return image
