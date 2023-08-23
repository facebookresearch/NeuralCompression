# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional

import torch

from neuralcompression.layers import NormalizeLatent, ProjectLatent, VQBottleneck
from neuralcompression.models import VqVaeXCiTAutoencoder

LOGGER = logging.getLogger(__file__)

VALID_WEIGHTS = [
    "vqvae_xcit_p8_ch64_cb1024_h8",
]


def _build_msillm_vqvae(vq_model: VqVaeXCiTAutoencoder, weights: Optional[str] = None):
    if weights is not None:
        if weights not in VALID_WEIGHTS:
            raise ValueError(
                f"Unrecognized weights {weights}, must be one of {VALID_WEIGHTS}"
            )

        url = (
            "https://dl.fbaipublicfiles.com/NeuralCompression/2023-muckley-msillm/"
            + f"{weights}.ckpt"
        )

        LOGGER.info(
            f"Using {weights} MS-ILLM VQ-VAE weights. These weights are "
            f"released under the CC-BY-NC 4.0 license which can be found at "
            "https://github.com/facebookresearch/NeuralCompression/tree/main/WEIGHTS_LICENSE."
        )

        vq_model.load_state_dict(
            torch.hub.load_state_dict_from_url(url, map_location="cpu")
        )

    return vq_model


def vqvae_xcit_p8_ch64_cb1024_h8(pretrained=False, **kwargs):
    if pretrained is True:
        weights = "vqvae_xcit_p8_ch64_cb1024_h8"
    else:
        weights = None

    vq_model = VqVaeXCiTAutoencoder(
        ch=64,
        out_ch=3,
        ch_mult=[1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[32],
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        resolution=256,
        embed_dim=256,
        freeze_encoder=False,
        norm_type="channel",
        bottleneck_op=ProjectLatent(
            input_dim=256,
            output_dim=8,
            child=NormalizeLatent(
                child=VQBottleneck(codebook_size=1024, vector_length=8)
            ),
        ),
    )

    return _build_msillm_vqvae(vq_model, weights=weights)
