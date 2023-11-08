# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from projects.torch_vct.neural.entropy_model_layers import (
    LearnedPosition,
    StartSym,
    TransformerBlock,
)


def test__shift_to_the_right():
    # create a sample input tensor with shape [B, C, seq_len]
    B, seq_length, hidden_dim = 2, 3, 4
    x = torch.randn(B, seq_length, hidden_dim)

    # shift x to the right and prefix it with zeros
    y = StartSym._shift_to_the_right(x)
    # check shapes
    assert y.shape == x.shape
    # check equality of elements
    assert torch.equal(y[:, 0, :], x.new_zeros(B, hidden_dim))
    assert torch.equal(y[:, 1:, :], x[:, :-1, :])
    # create a sample prefix tensor with shape [B, 1, hidden_dim] and prefix x
    prefix = torch.randn(B, 1, hidden_dim)
    y = StartSym._shift_to_the_right(x, prefix=prefix)
    # check shapes
    assert y.shape == x.shape
    # check equality of elements
    assert torch.equal(y[:, [0], :], prefix)
    assert torch.equal(y[:, 1:, :], x[:, :-1, :])


def test_startsym():
    # create a sample input tensor with shape [B, seq_len, hidden_dim] hidden_dim=C
    B, seq_len, hidden_dim = 2, 3, 4
    x = torch.randn(B, seq_len, hidden_dim)

    # create an instance of the StartSym class with num_channels=3
    start_sym = StartSym(hidden_dim=hidden_dim)

    # verify that the sym parameter has the correct shape and is initialized
    # with random uniform values in orrect domain
    assert start_sym.sym.shape == torch.Size([hidden_dim])
    assert start_sym.sym.min() > -3 and start_sym.sym.max() < 3.0

    # prefix x with the learned start symbol
    y = start_sym(x)

    # verify that y has the correct shape
    assert y.shape == x.shape
    # verify that y is prefixed with the learned start symbol
    assert all([torch.equal(y[i, 0, :], start_sym.sym) for i in range(y.shape[0])])
    assert torch.equal(y[:, 1:, :], x[:, :-1, :])
    # test the initial symbol is always the same
    assert all([(start_sym.sym == StartSym(hidden_dim).sym).all() for _ in range(3)])


def test_learned_position():
    seq_len, hidden_dim = 10, 8
    x = torch.randn(2, seq_len, hidden_dim)
    learned_position = LearnedPosition(seq_len, hidden_dim)

    # test shape
    assert learned_position(x).shape == x.shape

    # test that learned position encodings are added to the tensor
    y = learned_position(x)
    assert (y == x + learned_position._emb).all()


@pytest.mark.parametrize(
    "num_heads, hidden_dim, mlp_dim", [(4, 256, 512), (8, 512, 1024), (16, 1024, 2048)]
)
def test_transformerblock(num_heads, hidden_dim, mlp_dim):
    batch_size = 2
    seq_length = 16
    # create an instance of the TransformerBlock
    transformer_block = TransformerBlock(
        seq_length=seq_length,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        drop_out_rate=0.1,
        drop_path_rate=0.1,
        is_decoder=False,
    )
    input = torch.randn(batch_size, seq_length, hidden_dim)
    output = transformer_block(input)
    # assert that the output has the expected shape
    assert output.shape == torch.Size([batch_size, seq_length, hidden_dim])
