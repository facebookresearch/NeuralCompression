# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from projects.torch_vct.neural import layers_utils as lu


def test_mlp():
    batch_size = 32
    seq_len = 16
    in_features = 128
    mlp_dim = 256
    dropout = 0.1

    input = torch.randn(batch_size, seq_len, in_features)

    mlp = lu.MLP(in_features, mlp_dim, dropout)

    # Test forward pass
    output = mlp(input)
    assert output.shape == (batch_size, seq_len, in_features)

    # Test invalid input shape
    invalid_input = torch.randn(batch_size, in_features * 2)
    with pytest.raises(RuntimeError):
        mlp(invalid_input)


def test_make_embedding_layer():
    input_dim = 4
    hidden_dim = 8
    embedding_layer = lu.make_embedding(input_dim, hidden_dim)
    scale = 1 / torch.sqrt(torch.tensor(input_dim))

    assert isinstance(embedding_layer, torch.nn.Linear)
    assert embedding_layer.in_features == input_dim
    assert embedding_layer.out_features == hidden_dim

    # Mostly sanity checks
    assert isinstance(embedding_layer.weight, torch.Tensor)
    assert (embedding_layer.weight < scale).all()
    assert (embedding_layer.weight > -scale).all()
    assert isinstance(embedding_layer.bias, torch.Tensor)
    assert (embedding_layer.bias < scale).all()
    assert (embedding_layer.bias > -scale).all()


@pytest.mark.parametrize(
    "hidden_dim, num_heads, seq_len_q, seq_len_kv",
    [(32, 4, 4, 8), (32, 4, 4, 4), (32, 1, 4, 8), (32, 1, 8, 8)],
)
def test_windowmultiheadattention(hidden_dim, num_heads, seq_len_q, seq_len_kv):
    B = 2
    query = torch.randn(B, seq_len_q, hidden_dim)
    key = torch.randn(B, seq_len_kv, hidden_dim)
    value = torch.randn(B, seq_len_kv, hidden_dim)

    attn_drop = 0.1
    proj_drop = 0.1
    attn = lu.WindowMultiHeadAttention(hidden_dim, num_heads, attn_drop, proj_drop)

    # Test forward pass
    output, attention = attn(query, key, value)
    assert output.shape == (B, seq_len_q, hidden_dim)
    assert attention.shape == (B, num_heads, seq_len_q, seq_len_kv)

    # Test masked attention
    mask = torch.triu(torch.ones(seq_len_q, seq_len_q), 1)
    output, attention = attn(query, key, value, mask)
    assert output.shape == (B, seq_len_q, hidden_dim)
    assert attention.shape == (B, num_heads, seq_len_q, seq_len_kv)

    if seq_len_q == seq_len_kv:
        assert (attention == attention * (1 - mask)).all()
    else:
        keeps = 1.0 - torch.tile(mask, [1, seq_len_kv // seq_len_q])
        assert (attention == attention * keeps).all()

    # Test invalid inputs:
    # Invalid hidden dim
    with pytest.raises(ValueError):
        attn = lu.WindowMultiHeadAttention(
            hidden_dim, num_heads * 3, attn_drop, proj_drop
        )
    # Invalid seq_len_q, does not divide seq_len_kv
    with pytest.raises(AssertionError):
        attn(
            torch.randn(B, seq_len_q, hidden_dim),
            torch.randn(B, seq_len_q - 1, hidden_dim),
            torch.randn(B, seq_len_q - 1, hidden_dim),
        )
