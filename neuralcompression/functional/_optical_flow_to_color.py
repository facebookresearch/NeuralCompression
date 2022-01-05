# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from torch import Tensor

from ._hsv_to_rgb import hsv_to_rgb


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

    return hsv_to_rgb(hsv)
