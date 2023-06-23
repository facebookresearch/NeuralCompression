# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import replace

import torch

from neuralcompression import VqVaeAutoencoderOutput

from ._q_bottleneck import QBottleneck


class VQBottleneck(QBottleneck):
    """
    Vector quantization bottleneck.

    This class quantizes its input vectors based on nearest Euclidean distance.
    It also returns an estimate of the coding cost after the quantization
    operation.

    Args:
        codebook_size: The size of the codebook.
        vector_length: The length for each input vector.
        init_distribution: The initialization distribution for the codebook.
    """

    def __init__(
        self,
        codebook_size: int,
        vector_length: int,
        init_distribution: str = "uniform",
    ):
        self.vector_length = vector_length
        self.init_distribution = init_distribution

        super().__init__(
            codebook_sizes=[codebook_size],
            hidden_dims=[vector_length],
            init_distribution=init_distribution,
        )

    def forward(self, output: VqVaeAutoencoderOutput) -> VqVaeAutoencoderOutput:
        if output.latent is None:
            raise ValueError("Vector for quantization must be in latent attribute.")

        num_bytes = int(
            output.latent.shape[0]
            * output.latent.shape[-2]
            * output.latent.shape[-1]
            * math.log2(self.codebooks[0].shape[0])  # number of bits
            / 8
        )

        vq_output = self.quantize_vectors(
            preq_latents=output.latent,
            codebook=self.codebooks[0],
            probabilistic=False,
        )

        getattr(self, f"codebook_hits_{0}").index_add_(
            0, vq_output.indices, torch.ones_like(vq_output.indices, dtype=torch.long)
        )

        return replace(
            output,
            latent=vq_output.latent,
            prequantized_latent=vq_output.prequantized_latent,
            commitment_loss=vq_output.commitment_loss,
            embedding_loss=vq_output.embedding_loss,
            num_bytes=num_bytes,
            quantize_distances=vq_output.distances,
            indices=vq_output.indices,
        )
