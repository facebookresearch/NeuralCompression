# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from torch import Tensor

from neuralcompression import VqVaeAutoencoderOutput

from ._vqvae_xcit_autoencoder import VqVaeXCiTAutoencoder


class VqVaeProjector(nn.Module):
    """
    VQ-VAE Projector model

    This is intended to be used as a continuous-to-discrete mapping for
    adversarial optimization as described in the following paper:

    Improving Statistical Fidelity for Neural Image Compression with Implicit
    Local Likelihood Models
    MJ Muckley, A El-Nouby, K Ullrich, H Jégou, J Verbeek

    Args:
        vq_model: A VQ-VAE model with XCiT attention.
    """

    def __init__(self, vq_model: VqVaeXCiTAutoencoder):
        super().__init__()

        assert vq_model.bottleneck_op is not None
        self.encoder = vq_model.encoder
        self.bottleneck_op = vq_model.bottleneck_op
        self.eval()

        for param in self.parameters():
            param.requires_grad_(False)

    def train(self, mode: bool = True) -> "VqVaeProjector":
        """this should always be used in eval mode."""
        return super().train(False)

    def forward(self, image: Tensor) -> Tensor:
        output = self.bottleneck_op(
            VqVaeAutoencoderOutput(latent=self.encoder(image))
        ).indices
        assert output is not None

        return output
