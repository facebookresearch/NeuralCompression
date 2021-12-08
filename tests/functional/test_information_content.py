# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy
import pytest
import torch

from neuralcompression.functional import information_content


@pytest.mark.parametrize(
    "shape, seed",
    [
        ((3, 72, 64, 5), 0),
        ((5, 55, 18, 2), 1),
        ((6, 73, 35, 10), 2),
    ],
)
def test_information_content(shape, seed):
    # test all reductions of coding cost
    # also check base-2 and base-10
    rng = numpy.random.default_rng(seed)
    probabilities = torch.tensor(rng.uniform(size=shape))

    def batch_el_reduction(x):
        return torch.stack([torch.sum(el) for el in x])

    base_ops = {2: torch.log2, 10: torch.log10}
    reduction_ops = {
        "sum": torch.sum,
        "batch_el": batch_el_reduction,
        "none": lambda x: x,
    }

    for base in (2, 10):
        for reduction in ("sum", "batch_el", "none"):
            torch_cost = -1 * reduction_ops[reduction](base_ops[base](probabilities))
            assert torch.allclose(
                information_content(probabilities, reduction=reduction, base=base),
                torch_cost,
            )
