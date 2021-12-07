# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def _create_dense_warp_base_grid(
    dim1: int,
    dim2: int,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Basic wrapper for meshgrid."""
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device("cpu")

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, dim1, dtype=dtype, device=device),
        torch.linspace(-1, 1, dim2, dtype=dtype, device=device),
    )

    # for gridding we need to flip the order here
    base_grid = torch.stack((grid_x, grid_y), dim=-1)

    return base_grid.unsqueeze(0)


def dense_image_warp(
    image: Tensor,
    flow: Tensor,
    mode: str = "bilinear",
    padding_mode: str = "border",
    align_corners: bool = False,
) -> Tensor:
    r"""
    Warp image based on flow grid.

    Designed to mimic behavior of ``tf.contrib.image.dense_image_warp``. This
    function uses ``torch.nn.functional.grid_sample`` as its interpolation
    backend.

    * This function essentially applies inverse optical flow, i.e., for an
      image function :math:`f(x)`, we compute :math:`f(x+\delta)`, where
      :math:`\delta` is the flow. The flow uses the normalized grid in
      ``[-1, 1]`` as detailed in ``torch.nn.functional.grid_sample``. See
      ``torch.nn.functional.grid_sample`` for details.

    Args:
        image: Input image to be warped.
        flow: Optical flow field for applying warp. Can be different size than
            ``image``.
        mode: Interpolation mode to calculate output values ``'bilinear'`` |
            ``'nearest'`` | ``'bicubic'``. Default: ``'bilinear'``.
        padding_mode: Padding mode for outside grid values ``'zeros'`` |
            ``'border'`` | ``'reflection'``.
        align_corners: Whether to align corners. See
            ``torch.nn.functional.grid_sample``.

    Returns:
        The warped image.
    """
    if (not image.dtype == flow.dtype) or (not image.device == flow.device):
        raise ValueError("Either dtype or device not matched between inputs.")

    if not flow.shape[-1] == 2:
        raise ValueError("dense_image_warp only implemented for 2D images.")

    base_grid = _create_dense_warp_base_grid(
        flow.shape[1], flow.shape[2], dtype=image.dtype, device=image.device
    ).repeat(flow.shape[0], 1, 1, 1)

    return F.grid_sample(
        image,
        base_grid + flow,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
