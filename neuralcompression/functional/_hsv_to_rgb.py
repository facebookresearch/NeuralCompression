# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor


def hsv_to_rgb(hsv: Tensor) -> Tensor:
    """
    Convert from HSV colorspace to RGB.

    Args:
        hsv: A 4-dimensional tensor in hue, saturation, vue colorspace.

    Returns:
        A f-dimensional tensor in red, green, blue colorspace.
    """
    if not hsv.ndim == 4:
        raise ValueError("Only implemented for 4-dimensional tensors.")
    if not hsv.shape[1] == 3:
        raise ValueError("Expected channel dimension to be of size 3 for H, S, V.")
    if hsv[:, 0].min() < 0.0 or hsv[:, 0].max() > 360.0:
        raise ValueError("Hue angle not in expected 0 to 360 degree range.")
    if hsv[:, 1:].min() < 0.0 or hsv[:, 1:].max() > 1.0:
        raise ValueError("Saturation or vue outside expected 0.0 to 1.0 range.")

    # from Wikipedia, hsv to rgb alternative

    def convert_fn(n):
        k = torch.fmod(n + hsv[:, 0] / 60, 6)
        return hsv[:, 2] - hsv[:, 2] * hsv[:, 1] * torch.max(
            torch.zeros_like(k), torch.min(torch.min(k, 4 - k), torch.ones_like(k))
        )

    return torch.stack((convert_fn(5), convert_fn(3), convert_fn(1)), dim=1)
