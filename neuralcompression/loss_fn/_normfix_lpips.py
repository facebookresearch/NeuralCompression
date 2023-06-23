# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from lpips import LPIPS
from lpips.lpips import spatial_average, upsample


def normalize_tensor(x, eps=1e-8):
    norm_factor = torch.sqrt(torch.sum(x**2 + eps, dim=1, keepdim=True))
    return x / (norm_factor)


class NormFixLPIPS(LPIPS):
    """
    LPIPS loss function for propagating gradients.

    The online implementation of LPIPS includes a square root normalization
    term that is often 0 during training. For the metric version of LPIPS, this
    is not an issue since the metric includes divide-by-0 protection. However,
    when using LPIPS as a loss function, the 0 can lead to infinite gradients.

    This class addresses this by putting the 0 protection inside the square
    root term, making gradient propagation stable.

    See parent class ``LPIPS`` for arguments.
    """

    def train(self, mode: bool = True) -> "NormFixLPIPS":
        """the lpips network should be in eval mode."""
        return super().train(False)

    def forward(self, in0, in1, retPerLayer=False, normalize=False):
        if (
            normalize
        ):  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1

        # v0.0 - original release had a bug, where input was not scaled
        in0_input, in1_input = (
            (self.scaling_layer(in0), self.scaling_layer(in1))
            if self.version == "0.1"
            else (in0, in1)
        )
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(
                outs1[kk]
            )
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        if self.lpips:
            if self.spatial:
                res = [
                    upsample(self.lins[kk](diffs[kk]), out_HW=in0.shape[2:])
                    for kk in range(self.L)
                ]
            else:
                res = [
                    spatial_average(self.lins[kk](diffs[kk]), keepdim=True)
                    for kk in range(self.L)
                ]
        else:
            if self.spatial:
                res = [
                    upsample(diffs[kk].sum(dim=1, keepdim=True), out_HW=in0.shape[2:])
                    for kk in range(self.L)
                ]
            else:
                res = [
                    spatial_average(diffs[kk].sum(dim=1, keepdim=True), keepdim=True)
                    for kk in range(self.L)
                ]

        val = 0
        for lind in range(self.L):
            val += res[lind]

        if retPerLayer:
            return (val, res)
        else:
            return val
