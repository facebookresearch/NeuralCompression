"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import numpy as np
import torch


def create_input(shape):
    x = np.arange(np.product(shape)).reshape(shape)

    return torch.from_numpy(x).to(torch.get_default_dtype())
