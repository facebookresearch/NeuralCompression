# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


def make_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def make_deconv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
) -> nn.ConvTranspose2d:
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )


def make_embedding(input_dim: int, hidden_dim: int) -> nn.Module:
    """
    Constructs and returns an embedding layer, which is a simple (dense) linear layer

    Args:
        input_dim: input dimensions (input to linear layer)
        hidden_dim: output size of the linear layer

    Returns:
        a linear nn.Module, initialized with random uniform weights and biases
    """
    scale = 1 / input_dim**0.5
    linear = torch.nn.Linear(input_dim, hidden_dim, bias=True)
    # initialise weights in the same was as vct
    torch.nn.init.uniform_(linear.weight, -scale, scale)
    torch.nn.init.uniform_(linear.bias, -scale, scale)
    return linear


def init_weights_truncated_normal(m) -> None:
    """
    Initialise weights with truncated normal.
    Weights that fall outside 2 stds are resampled.
    See torch.nn.init.trunc_normal_ for details.

    Args:
        m: weights

    Examples:
    >>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    >>> net.apply(init_weights_truncated_normal)
    """
    std = 0.02
    if isinstance(m, nn.Linear):
        torch.nn.init.trunc_normal_(m.weight, std=std, a=-2 * std, b=2 * std)
        m.bias.data.fill_(0.01)


class MLP(nn.Module):
    """MLP head for transformer blocks"""

    def __init__(
        self,
        in_features: int,
        mlp_dim: int,
        dropout: float,
    ) -> None:
        """
        MLP head for transformer blocks
        Args:
            expansion_rate: rate at which the input tensor is expanded
            dropout_rate: dropout rate
            input_shape: shape of the input tensor, with the last dimension as the
                size of channel -- [N1, ... Nn, C]
        """
        super().__init__()

        # Initialize linear layers with truncated normal
        self.fc1 = nn.Linear(in_features=in_features, out_features=mlp_dim)
        init_weights_truncated_normal(self.fc1)
        self.fc2 = torch.nn.Linear(in_features=mlp_dim, out_features=in_features)
        init_weights_truncated_normal(self.fc2)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Args:
            features: tensor of shape (batch_size, seq_len, hidden_dim)

        Returns:
            tensor of shape (batch_size, seq_len, hidden_dim)
        """
        input = self.fc1(input)
        input = self.act(input)
        input = self.dropout(input)
        input = self.fc2(input)
        return self.dropout(input)


class WindowMultiHeadAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """Windowed multi-head self-attention

        Args:
            hidden_dim: size of the hidden units
            num_heads: number of attention heads
            attn_drop: dropout rate of the attention layer. Defaults to 0.0.
            proj_drop: dropout rate of the projection layer. Defaults to 0.0.

        Raises:
            ValueError: if `hidden_dim` is not a multiple of `num_heads`
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"Size of hidden units ({hidden_dim}) not divisible by number "
                f"of heads ({num_heads})."
            )
        # constnat to scale output before softmax-ing
        self.attn_scale = (hidden_dim // num_heads) ** (-0.5)
        # All linear layers are initialized with truncated normal
        self.q_linear = torch.nn.Linear(
            in_features=hidden_dim, out_features=hidden_dim, bias=True
        )
        init_weights_truncated_normal(self.q_linear)
        self.k_linear = torch.nn.Linear(
            in_features=hidden_dim, out_features=hidden_dim, bias=True
        )
        init_weights_truncated_normal(self.k_linear)
        self.v_linear = torch.nn.Linear(
            in_features=hidden_dim, out_features=hidden_dim, bias=True
        )
        init_weights_truncated_normal(self.v_linear)
        self.attn_dropout = nn.Dropout(p=attn_drop)
        self.proj = torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        init_weights_truncated_normal(self.proj)
        self.proj_dropout = torch.nn.Dropout(p=proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)  # -1 is the default in tf

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Compute the windowed multi-head self-attention.

        Note: seq_length of keys and values (`seq_len_kv`) must be an integer
            multiple of the seq_length of the query (`seq_len_q`).

        Args:
            query: tensor of shape (b', seq_len_q, hidden_dim),
                representing the query
            key: tensor of shape (b', seq_len_kv, hidden_dim),
                representing the key
            value: tensor of shape (b', seq_len_kv, hidden_dim),
                representing the value
            mask: optional tensor of shape (b', seq_len_q, seq_len_q),
                representing the mask to apply to the attention; 1s will be masked
            -> b' is an augmented batch size that includes the total number of patches
            -> by default, hidden_dim is 768

        Returns:
            tensor of feautures of shape (b', seq_len_q, hidden_dim), as well as the
            attention matrix used, shape (b', num_heads, seq_len_q, seq_len_kv).
        """
        *b, seq_len_q, c = query.shape
        assert c == self.hidden_dim, f"Shape mismatch, {c} != {self.hidden_dim}"
        seq_len_kv = value.shape[-2]
        assert seq_len_kv % seq_len_q == 0, (
            f"seq_length of keys and values = {seq_len_kv}  must be an integer multiple "
            f"of the seq_length of the query = {seq_len_q}"
        )
        blowup = seq_len_kv // seq_len_q

        query = self.q_linear(query)  # [b', seq_len_q, hidden_dim]
        key = self.k_linear(key)  # [b', seq_len_kv, hidden_dim]
        value = self.v_linear(value)  # [b', seq_len_kv, hidden_dim]

        # reshape by splitting channels into num_heads, then permute:
        query = (
            query.reshape(*b, seq_len_q, self.num_heads, c // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # [b', num_heads, seq_len_q, c // num_heads]
        key = (
            key.reshape(*b, seq_len_kv, self.num_heads, c // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # [b', num_heads, seq_len_kv, c // num_heads]
        value = (
            value.reshape(*b, seq_len_kv, self.num_heads, c // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # [b', num_heads, seq_len_kv, c // num_heads]

        # b', num_heads, seq_len_q, seq_len_kv
        attn = (
            torch.matmul(
                query,  # [b', num_heads, seq_len_q, c // num_heads]
                key.transpose(2, 3),  # [b', num_heads, c // num_heads, seq_len_kv]
            )  # [b', num_heads,seq_len_q, seq_len_kv]
            * self.attn_scale
        )

        if mask is not None:
            if mask.shape[-2:] != (seq_len_q, seq_len_q):
                # Note that we only mask for self-attention in the decoder,
                # where the attention matrix has shape (..., seq_len_q, seq_len_q).
                raise ValueError(f"Invalid mask shape: {mask.shape}.")

            # Here, we add the mask to the attention with a large negative multiplier,
            # as this goes into a softmax it will become 0
            tile_pattern = [1] * mask.dim()
            tile_pattern[-1] = blowup
            attn = attn + torch.tile(mask, tile_pattern) * -1e6
        else:
            tile_pattern = None

        attn = self.softmax(attn)  # [b', num_heads, seq_length_q, seq_length_kv]

        if mask is not None and tile_pattern is not None:
            # We use the mask again, to be ensure that no masked dimension
            # affects the output.
            keep = 1 - mask
            attn = attn * torch.tile(keep, tile_pattern)

        attn = self.attn_dropout(attn)  # [b', num_heads, seq_length_q, seq_length_kv]

        features = torch.matmul(
            attn, value
        )  # [b', num_heads, seq_len_q, d_model//num_heads], c=d_model
        assert features.shape == (*b, self.num_heads, seq_len_q, c // self.num_heads)

        features = (
            features.permute(0, 2, 1, 3)  # switch num_heads, seq_len_q dimensions.
            .contiguous()
            .reshape(*b, -1, self.hidden_dim)  # flatten num_heads, seq_len_q dimensions
        )
        features = self.proj(features)
        features = self.proj_dropout(features)
        assert features.shape == (*b, seq_len_q, c)
        return features, attn


class StochasticDepth(nn.Module):
    """Creates a stochastic depth layer."""

    def __init__(self, stochastic_depth_drop_rate: float) -> None:
        """Initializes a stochastic depth layer.

        Args:
          stochastic_depth_drop_rate: A `float` of drop rate.

        Returns:
          a tensor of the same shape as input.
        """
        super().__init__()
        self._drop_rate = stochastic_depth_drop_rate

    def forward(self, input: Tensor) -> Tensor:
        if not self.training or self._drop_rate == 0.0:
            return input

        keep_prob = 1.0 - self._drop_rate
        batch_size = input.shape[0]
        random_tensor = keep_prob
        random_tensor = random_tensor + torch.rand(
            [batch_size] + [1] * (input.dim() - 1), dtype=input.dtype
        )
        binary_tensor = torch.floor(random_tensor)
        return input / keep_prob * binary_tensor
