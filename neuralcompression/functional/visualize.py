"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import math

import torch
from torch import Tensor


def hsv2rgb(hsv: Tensor) -> Tensor:
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


def optical_flow_to_color(flow: Tensor, eps: float = 1e-5) -> Tensor:
    """
    Convert optical flow field to color map via HSV colorspace.

    Args:
        flow: An optical flow field to visualize.
        eps: Tolerance for magnitude normalization.

    Returns:
        A batch of 3-channel RGB images for visualization.
    """
    if not flow.ndim == 4:
        raise ValueError("Expected 4-dimensional input.")

    hsv = torch.zeros(
        (flow.shape[0], 3, flow.shape[2], flow.shape[3]),
        dtype=flow.dtype,
        device=flow.device,
    )

    def angles_in_0_to_360_range(yval, xval):
        angles = torch.atan2(yval, xval)  # returns -pi to pi
        angles[angles < 0] = 2 * math.pi + angles[angles < 0]  # convert to 0 to 2pi
        return (180 / math.pi) * angles

    hsv[:, 0] = angles_in_0_to_360_range(flow[:, 0], flow[:, 1])
    hsv[:, 1] = 1.0
    mag = torch.sqrt(flow[:, 0] ** 2 + flow[:, 1] ** 2)
    mag = mag - mag.min()
    mag = mag / (mag.max() + eps)
    hsv[:, 2] = mag

    return hsv2rgb(hsv)
