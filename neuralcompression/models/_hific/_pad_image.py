"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import typing

import torch
import torch.nn.functional


def _pad_image(
    image: torch.Tensor,
    dimensions: torch.Tensor,
    factor: typing.Union[int, typing.Tuple[int, int]],
) -> torch.Tensor:
    if isinstance(factor, int):
        factor_h = factor
        factor_w = factor_h
    else:
        factor_h, factor_w = factor

    return torch.nn.functional.pad(
        image,
        (
            0,
            ((factor_w - (dimensions[1] % factor_w)) % factor_w),
            0,
            ((factor_h - (dimensions[0] % factor_h)) % factor_h),
        ),
        mode="reflect",
    )
