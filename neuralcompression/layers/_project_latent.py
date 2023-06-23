# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch.nn as nn

from neuralcompression import VqVaeAutoencoderOutput


class ProjectLatent(nn.Module):
    """
    Applies a convolution projection of the latent.

    This class is intended to wrap a recursive tree of operations. The
    projection is applied to the input, followed by whatever operation is
    contained in ``child``. The output of ``child`` then has an inverse
    projection applied.

    Args:
        input_dim: The input dimension for the projection.
        output_dim: The output dimension for the projection.
        child: A child operation to apply between forward and inverse
            projections.
    """

    def __init__(
        self, input_dim: int, output_dim: int, child: Optional[nn.Module] = None
    ):
        super().__init__()
        self.child = child
        self.input_conv = nn.Conv2d(
            in_channels=input_dim, out_channels=output_dim, kernel_size=1
        )
        self.output_conv = nn.Conv2d(
            in_channels=output_dim, out_channels=input_dim, kernel_size=1
        )

    def forward(self, output: VqVaeAutoencoderOutput) -> VqVaeAutoencoderOutput:
        if self.child is not None:
            output.latent = self.input_conv(output.latent)
            output = self.child(output)
            output.latent = self.output_conv(output.latent)
            return output
        else:
            output.latent = self.output_conv(self.input_conv(output.latent))
            return output
