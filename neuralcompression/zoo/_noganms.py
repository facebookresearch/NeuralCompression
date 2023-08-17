# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch

from neuralcompression.models import HiFiCAutoencoder

VALID_WEIGHTS = [
    "target_0.035bpp",
    "target_0.07bpp",
    "target_0.14bpp",
    "target_0.3bpp",
    "target_0.45bpp",
    "target_0.9bpp",
]

QUALITIES = [1, 2, 3, 4, 5, 6]


def _build_noganms(weights: Optional[str] = None):
    if weights not in VALID_WEIGHTS:
        raise ValueError(
            f"Unrecognized weights {weights}, must be one of {VALID_WEIGHTS}"
        )

    model = HiFiCAutoencoder()
    if weights is not None:
        url = (
            "https://dl.fbaipublicfiles.com/NeuralCompression/2023-muckley-msillm/"
            + f"noganms_{weights}.ckpt"
        )

        model.load_state_dict(torch.hub.load_state_dict_from_url(url))

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
        weights = "target_0.455bpp"
    else:
        weights = None

    return _build_noganms(weights=weights)


def noganms_quality_6(pretrained=False, **kwargs):
    if pretrained is True:
        weights = "target_0.9bpp"
    else:
        weights = None

    return _build_noganms(weights=weights)
