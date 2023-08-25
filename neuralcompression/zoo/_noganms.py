# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional

import torch

from neuralcompression.models import HiFiCAutoencoder

LOGGER = logging.getLogger(__file__)

VALID_WEIGHTS = [
    "target_0.035bpp",
    "target_0.07bpp",
    "target_0.14bpp",
    "target_0.3bpp",
    "target_0.45bpp",
    "target_0.9bpp",
]


def _build_noganms(weights: Optional[str] = None):
    model = HiFiCAutoencoder()
    if weights is not None:
        if weights not in VALID_WEIGHTS:
            raise ValueError(
                f"Unrecognized weights {weights}, must be one of {VALID_WEIGHTS}"
            )

        url = (
            "https://dl.fbaipublicfiles.com/NeuralCompression/2023-muckley-msillm/"
            + f"noganms_{weights}.ckpt"
        )

        LOGGER.info(
            f"Using {weights} NoGAN-MS weights. These weights are released under "
            f"the CC-BY-NC 4.0 license which can be found at "
            "https://github.com/facebookresearch/NeuralCompression/tree/main/WEIGHTS_LICENSE."
        )

        model.load_state_dict(
            torch.hub.load_state_dict_from_url(url, map_location="cpu")
        )

    return model


def noganms_quality_1(pretrained=False, **kwargs):
    if pretrained is True:
        weights = "target_0.035bpp"
    else:
        weights = None

    return _build_noganms(weights=weights)


def noganms_quality_2(pretrained=False, **kwargs):
    if pretrained is True:
        weights = "target_0.07bpp"
    else:
        weights = None

    return _build_noganms(weights=weights)


def noganms_quality_3(pretrained=False, **kwargs):
    if pretrained is True:
        weights = "target_0.14bpp"
    else:
        weights = None

    return _build_noganms(weights=weights)


def noganms_quality_4(pretrained=False, **kwargs):
    if pretrained is True:
        weights = "target_0.3bpp"
    else:
        weights = None

    return _build_noganms(weights=weights)


def noganms_quality_5(pretrained=False, **kwargs):
    if pretrained is True:
        weights = "target_0.45bpp"
    else:
        weights = None

    return _build_noganms(weights=weights)


def noganms_quality_6(pretrained=False, **kwargs):
    if pretrained is True:
        weights = "target_0.9bpp"
    else:
        weights = None

    return _build_noganms(weights=weights)
