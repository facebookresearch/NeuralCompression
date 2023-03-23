# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .layers_utils import MLP, StochasticDepth, WindowMultiHeadAttention


class LearnedPosition(nn.Module):
    """
    Learned poisitional encoding
    """

    def __init__(self, seq_length: int, hidden_dim: int) -> None:
        """
        Learned positional encoding

        Args:
            seq_length: sequence length.
            hidden_dim: hidden (model) dimension
        """
        super().__init__()
        self._emb = torch.nn.parameter.Parameter(
            torch.empty(1, seq_length, hidden_dim).normal_(std=0.02)
        )  # [1, seq_length, hidden_dim]
        self._seq_len = seq_length
        self._hidden_dim = hidden_dim

    def forward(self, x: Tensor) -> Tensor:
        """Adds positional encodings to an input

        Args:
            x: tensor to which the positional encodings will be added,
                expected shape is [B, seq_len, hidden_dim]

        Returns:
            the input tensor with the positional encodings added.
        """
        assert x.dim() == 3 and x.shape[1:] == torch.Size(
            [self._seq_len, self._hidden_dim]
        ), f"Expected [B, seq_length, hidden_dim] got {x.shape}"
        return x + self._emb


class StartSym(nn.Module):
    """Helper to learn a "zero" symbol, i.e., the first symbol to feed."""

    def __init__(self, hidden_dim: int) -> None:
        """
        Learn a zero-th symbol of dim `hidden_dim` (ie channels).

        Args:
            hidden_dim: number of input channels / hidden dimension
        """
        super().__init__()

        def initializer(param):
            # for reporoducibility we fix the seeed
            generator = torch.Generator().manual_seed(42)
            return param.uniform_(-3.0, 3.0, generator=generator)

        self.sym = nn.parameter.Parameter(torch.empty(hidden_dim))  # [hidden_dim]
        self.sym.data = initializer(self.sym.data)

    @staticmethod
    def _shift_to_the_right(x: Tensor, prefix: Optional[Tensor] = None) -> Tensor:
        """
        Shifts the input tensor `x` to the right across the seq_len (second) dimension
            and returns [pad, x[..., :-1]]

        Args:
            x: tensor, expected shape [B, seq_length, hidden_dim]
            prefix: the tensor to prefix to `x` with  shape [B, 1, hidden_dim].
                If `prefix` is None, it is set to zeros.

        Returns:
            the input tensor `x` shifted to the right by one position and prefixed
            with the specified `prefix`; shape is the same as `x`
        """
        B, seq_len, hidden_dim = x.shape
        expected_prefix_shape = (B, 1, hidden_dim)
        if prefix is None:
            prefix = x.new_zeros(expected_prefix_shape)
        assert prefix.shape == expected_prefix_shape, "shape mismatch!"
        return torch.cat([prefix, x[:, :-1, :]], dim=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Prefixes an input tensor with the learned start symbol. In other words, the
            first symbol in the sequence will be substituted by the learnt symbol.

        Args:
            x: tensor to prefix, expected shape is [B, seq_len, hidden_dim]

        Returns:
            a tensor prefixed with the learned start symbol, i.e. learnt symbol is the
                0th symbol in the sequence.
        """
        B, seq_len, hidden_dim = x.shape
        # [hidden_dim] * [1, hidden_dim]
        prefix = self.sym * x.new_ones(B, 1, hidden_dim)  # [B, 1, hidden_dim]
        return self._shift_to_the_right(x, prefix)


class TransformerBlock(nn.Module):
    """
    Single transformer block, can be used for both encoder and decoder
    """

    def __init__(
        self,
        seq_length: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        drop_out_rate: float = 0.1,
        drop_path_rate: float = 0.1,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        is_decoder: bool = False,
    ) -> None:
        super().__init__()
        if is_decoder:
            self.register_buffer(
                "look_ahead_mask",
                torch.triu(
                    torch.ones(seq_length, seq_length, dtype=torch.float32), diagonal=1
                ),
            )
        else:
            self.look_ahead_mask = None
        self.is_decoder = is_decoder
        self.num_heads = num_heads

        # --- BLOCK 1 ---
        self.ln_1a = norm_layer(hidden_dim)
        self.self_attention = WindowMultiHeadAttention(
            hidden_dim,
            num_heads,
            attn_drop=drop_out_rate,
            proj_drop=drop_out_rate,
        )
        self.ln_1b = norm_layer(hidden_dim)
        self.mlp1 = MLP(
            in_features=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=drop_out_rate,
        )

        # --- BLOCK 2 ---
        self.ln_2a = norm_layer(hidden_dim)
        self.cross_attention = WindowMultiHeadAttention(
            hidden_dim,
            num_heads,
            attn_drop=drop_out_rate,
            proj_drop=drop_out_rate,
        )
        self.ln_2b = norm_layer(hidden_dim)
        self.mlp2 = MLP(in_features=hidden_dim, mlp_dim=mlp_dim, dropout=drop_out_rate)

        self.drop_path = StochasticDepth(drop_path_rate)

    def forward(
        self,
        input: Tensor,
        encoder_output: Optional[Tensor] = None,
    ) -> Tensor:
        """_summary_

        Args:
            input: tensor with expected shape [B, seq_len, C]
            encoder_output: output from the encoder, used in decoder only.
                Defaults to None.

        Returns:
            tensor of the same shape of `input`
        """
        assert (
            input.dim() == 3
        ), f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}"
        if self.is_decoder and encoder_output is None:
            raise ValueError("Decoder needs `encoder_output`")

        # ------ FIRST BLOCK ------
        x = self.ln_1a(input)  # normalise input
        x, _ = self.self_attention(
            query=x, key=x, value=x, mask=self.look_ahead_mask
        )  # attend
        x = input + self.drop_path(x)  # skip connnection
        # normalise -> MLP -> add to itself == OUTPUT of FIRST BLOCK
        x = x + self.drop_path(self.mlp1(self.ln_1b(x)))  # [B, seq_len, hidden_dim]

        # ------ SECOND BLOCK ------
        y = self.ln_2a(x)  # normalise input, [B, seq_len, hidden_dim]
        y, _ = self.cross_attention(
            query=y,
            key=encoder_output if encoder_output is not None else y,
            value=encoder_output if encoder_output is not None else y,
            mask=None,
        )  # attend
        y = x + self.drop_path(y)  # skip connection
        # normalise -> MLP -> add to itself == OUTPUT of SECOND BLOCK
        y = y + self.drop_path(self.mlp2(self.ln_2b(y)))  # [B, seq_len, hidden_dim]
        return y


class Transformer(nn.Module):
    """
    Stack of transformer blocks
    """

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_expansion: int,
        dropout: float,
        is_decoder: bool = False,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ) -> None:
        super().__init__()
        self.is_decoder = is_decoder
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        # Stack num_layers transformer blocks:
        for i in range(num_layers):
            self.layers.add_module(
                f"encoder_layer_{i}",
                TransformerBlock(
                    seq_length=seq_length,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    mlp_dim=mlp_expansion * hidden_dim,
                    drop_out_rate=dropout,
                    drop_path_rate=dropout,
                    norm_layer=norm_layer,
                    is_decoder=is_decoder,
                ),
            )

    def forward(self, input: Tensor, encoder_output: Optional[Tensor]) -> Tensor:
        """
        In the case of a transformer Decoder: the forward pass predicts the distribution
        of the `latent` given `encoder_output`

        Args:
            input: tensor of shape [B', seq_length, hidden_dim]
            encoder_output: tensor of shape [B', seq_length_encoder, hidden_dim],
                which is the result of the encoder output (concatenated).
                Optional (not needed for encoder Transformer), defaults to None.

        Returns:
            tensor of shape [B', seq_length, hidden_dim]
        """
        assert (
            input.dim() == 3
        ), f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}"
        if encoder_output is not None:
            if input.shape[-1] != encoder_output.shape[-1]:
                raise ValueError(
                    f"Expected latent.shape[-1] == encoder_output.shape[-1], "
                    f"got {input.shape[-1]}, {encoder_output.shape[-1]}"
                )

        for layer in self.layers:
            input = layer(input=input, encoder_output=encoder_output)
        return input


class EncoderSection(Transformer):
    """
    A wrapper around `Transformer` turning it into an Encoder by setting the following:
        - is_decoder=False
        - seq_length=0
        - encoder_output=None
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_expansion: int,
        dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ) -> None:
        super().__init__(
            seq_length=0,  # NO-OP: used for look_ahead_mask only, no masking in encoder
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_expansion=mlp_expansion,
            dropout=dropout,
            is_decoder=False,
            norm_layer=norm_layer,
        )

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input, encoder_output=None)
