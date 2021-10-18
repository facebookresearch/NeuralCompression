"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from ._estimate_tails import estimate_tails
from ._log_ndtr import log_ndtr
from ._ndtr import ndtr
from ._pmf_to_quantized_cdf import pmf_to_quantized_cdf
from ._quantization_offset import quantization_offset
from ._soft_round import (
    soft_round,
    soft_round_conditional_mean,
    soft_round_inverse,
)
from .complexity import count_flops
from .distortion import (
    learned_perceptual_image_patch_similarity,
    multiscale_structural_similarity,
)
from .information import information_content
from .visualize import hsv2rgb, optical_flow_to_color
from .warp import dense_image_warp
