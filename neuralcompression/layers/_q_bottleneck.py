# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import NamedTuple, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _build_codebook(
    codebook_size: int, vector_length: int, init_distribution: str = "uniform"
) -> Tensor:
    if init_distribution == "uniform":
        codebook = torch.empty(codebook_size, vector_length).uniform_(
            -1.0 / codebook_size, 1.0 / vector_length
        )
    elif init_distribution == "normal":
        codebook = torch.randn(codebook_size, vector_length)
    else:
        raise ValueError(f"Unrecognized init_distribution {init_distribution}.")

    return codebook


class VQOutput(NamedTuple):
    latent: Tensor
    prequantized_latent: Tensor
    commitment_loss: Tensor
    embedding_loss: Tensor
    indices: Tensor
    num_bytes: Optional[int] = None
    distances: Optional[Tensor] = None


class QBottleneck(nn.Module):
    """
    Quantization bottleneck.

    This class quantizes its input based on a codebook. Each entry is quantized
    to its nearest codebook representation according to Euclidean distance.

    Args:
        codebook_sizes: Size of the codebook for each quantization level.
        hidden_dims: Hidden dimension size, i.e., the sizes of the input
            vectors.
        init_distribution: Initialization distribution for the codebooks.
        normalize_codebook: Whether to renormalize the codebook prior to each
            quantization run.
    """

    perplexities: Tensor
    max_hits: Tensor
    percentage_hits: Tensor

    def __init__(
        self,
        codebook_sizes: Sequence[int],
        hidden_dims: Sequence[int],
        init_distribution: str = "uniform",
        normalize_codebook: bool = True,
    ):
        super().__init__()
        self.normalize_codebook = normalize_codebook

        codebook_list = []
        for codebook_size, hidden_dim in zip(codebook_sizes, hidden_dims):
            codebook_list.append(
                nn.Parameter(
                    _build_codebook(
                        codebook_size,
                        hidden_dim,
                        init_distribution=init_distribution,
                    )
                )
            )

        self.codebooks = nn.ParameterList(codebook_list)
        num_codebooks = len(self.codebooks)

        # tracking for codebook metrics
        for ind, codebook in enumerate(self.codebooks):
            self.register_buffer(
                f"codebook_hits_{ind}", torch.zeros(codebook.shape[0], dtype=torch.long)
            )
        self.register_buffer("perplexities", torch.zeros(num_codebooks))
        self.register_buffer("max_hits", torch.zeros(num_codebooks))
        self.register_buffer("percentage_hits", torch.zeros(num_codebooks))

    def reset_hits(self):
        for codebook_hits in self.named_buffers(prefix="codebook_hits"):
            codebook_hits *= 0

    def update_metrics(self):
        assert len(self.percentage_hits) == len(self.max_hits) == len(self.perplexities)
        num_codebooks = len(self.percentage_hits)
        # codebook hit stats
        for level in range(num_codebooks):
            codebook_hits: Tensor = getattr(self, f"codebook_hits_{level}")
            self.percentage_hits[level] = (
                torch.sum(codebook_hits != 0) / codebook_hits.numel()
            )
            self.max_hits[level] = torch.max(codebook_hits)
            probs = codebook_hits / torch.sum(codebook_hits)
            self.perplexities[level] = 2 ** (
                -torch.sum(probs * torch.log2(probs + 1e-7))
            )

    def quantize_vectors(
        self,
        preq_latents: Tensor,
        codebook: Tensor,
        probabilistic: bool = False,
    ) -> VQOutput:
        batch_size, hidden_dim, num_y, num_x = preq_latents.shape
        latents = preq_latents.permute(0, 2, 3, 1).reshape(-1, hidden_dim)

        # vector is (num_vectors, vector_length), already normalized
        # codebook is (codebook_size, vector_length)
        if self.normalize_codebook:
            codebook = F.normalize(codebook, dim=1)

        # distances variable is (num_vectors, codebook_size)
        distances = torch.sum(
            (codebook.unsqueeze(0) - latents.unsqueeze(1)) ** 2, dim=2
        )

        if probabilistic:
            # create probability weighted by distance
            weights = 1 / (distances + 1e-7)
            cumulative = torch.cumsum(
                weights / torch.sum(weights, dim=1, keepdim=True), dim=1
            )
            # hacky way to find indices
            # pick approximate indices based on probabilities
            tmp = cumulative - torch.rand_like(cumulative[:, :1])
            tmp[tmp < 0] = 1.0
            indices = torch.argmin(tmp, dim=1)
        else:
            # return the shortest distance
            indices = torch.argmin(distances, dim=1)

        # pick out and reshape the latents
        latents = (
            codebook[indices]
            .view(batch_size, num_y, num_x, hidden_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        # VQ losses
        commitment_loss = F.mse_loss(latents.detach(), preq_latents)
        embedding_loss = F.mse_loss(latents, preq_latents.detach())

        return VQOutput(
            latent=preq_latents + (latents - preq_latents).detach(),  # STE
            prequantized_latent=preq_latents,
            commitment_loss=commitment_loss,
            embedding_loss=embedding_loss,
            indices=indices,
            distances=distances,
        )
