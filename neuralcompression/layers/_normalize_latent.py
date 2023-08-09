# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# TODO: Consider reimplementing these recursive structures as a Python list.

from typing import Optional

import torch.nn as nn
import torch.nn.functional as F

from neuralcompression import VqVaeAutoencoderOutput


class NormalizeLatent(nn.Module):
    """
    Latent normalization.

    This class applies Euclidean latent normalization along the second
    dimension prior to a subsequent child operation.

    Args:
        child: A child operation to follow normalization within an overall
            recursive tree.
    """

    def __init__(self, child: Optional[nn.Module] = None):
        super().__init__()
        self.child = child

    def forward(self, output: VqVaeAutoencoderOutput) -> VqVaeAutoencoderOutput:
        if output.latent is None:
            raise ValueError("Expected tensor value in latent field.")
        if self.child is not None:
            output.latent = F.normalize(output.latent, dim=1)
            return self.child(output)
        else:
            output.latent = F.normalize(output.latent, dim=1)
            return output
