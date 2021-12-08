# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import numpy
import pytest
import torch

from neuralcompression.functional import hsv_to_rgb


@pytest.mark.parametrize(
    "shape, seed", [((5, 3, 64, 64), 0), ((2, 3, 76, 55), 1), ((10, 3, 32, 64), 2)]
)
def test_hsv_to_rgb(shape, seed):
    # verify correctness of hsv2rgb function by comparing to OpenCV impl.
    rng = numpy.random.default_rng(seed)
    hsvs = rng.uniform(low=0.0, high=1.0, size=shape)
    hsvs[:, 0] *= 360.0
    hsvs = torch.tensor(hsvs, dtype=torch.float32)

    rgb_torch = hsv_to_rgb(hsvs)

    for i, hsv in enumerate(hsvs):
        bgr = numpy.flip(
            cv2.cvtColor(hsv.permute(1, 2, 0).numpy(), cv2.COLOR_HSV2BGR), 2
        )

        # low atol due to float32
        assert numpy.allclose(bgr, rgb_torch[i].permute(1, 2, 0).numpy(), atol=1e-6)
