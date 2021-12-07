# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from neuralcompression.layers import SimplifiedGDN, SimplifiedInverseGDN


@pytest.mark.parametrize(
    "shape",
    [([5, 3, 160, 160]), ([3, 6, 64, 64]), ([1, 12, 32, 32])],
)
def test_simplified_gdn(shape):
    gen = torch.Generator()
    gen.manual_seed(123)
    x = torch.randn(shape)

    layer = SimplifiedGDN(x.shape[1])
    output = layer(x)

    assert x.shape == output.shape


@pytest.mark.parametrize(
    "shape",
    [([5, 3, 160, 160]), ([3, 6, 64, 64]), ([1, 12, 32, 32])],
)
def test_simplified_gdn_clamp(shape):
    gen = torch.Generator()
    gen.manual_seed(123)
    x = torch.randn(shape)

    # floating precision error requires this test to be double
    layer = SimplifiedGDN(x.shape[1]).to(torch.double)
    params = layer.parameters()
    layer.gamma.data = layer.gamma.data - 5
    layer.beta.data = layer.beta.data - 5

    _ = layer(x.to(torch.double))

    # make sure we clamped the params
    assert torch.allclose(layer.gamma.data, torch.zeros_like(layer.gamma.data))
    assert torch.allclose(
        layer.beta.data, torch.ones_like(layer.beta.data) * layer.beta_min
    )

    # make sure we didn't delete parameters
    assert len(list(params)) == 2


@pytest.mark.parametrize(
    "shape",
    [([5, 3, 160, 160]), ([3, 6, 64, 64]), ([1, 12, 32, 32])],
)
def test_simplified_inverse_gdn(shape):
    gen = torch.Generator()
    gen.manual_seed(123)
    x = torch.randn(shape)

    layer = SimplifiedInverseGDN(x.shape[1])
    output = layer(x)

    assert x.shape == output.shape


@pytest.mark.parametrize(
    "shape",
    [([5, 3, 160, 160]), ([3, 6, 64, 64]), ([1, 12, 32, 32])],
)
def test_simplified_inverse_gdn_clamp(shape):
    gen = torch.Generator()
    gen.manual_seed(123)
    x = torch.randn(shape)

    # floating precision error requires this test to be double
    layer = SimplifiedInverseGDN(x.shape[1]).to(torch.double)
    params = layer.parameters()
    layer.gamma.data = layer.gamma.data - 5
    layer.beta.data = layer.beta.data - 5

    _ = layer(x.to(torch.double))

    # make sure we clamped the params
    assert torch.allclose(layer.gamma.data, torch.zeros_like(layer.gamma.data))
    assert torch.allclose(
        layer.beta.data, torch.ones_like(layer.beta.data) * layer.beta_min
    )

    # make sure we didn't delete parameters
    assert len(list(params)) == 2
