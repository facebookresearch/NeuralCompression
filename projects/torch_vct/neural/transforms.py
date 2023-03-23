# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

## This module contains modules implementing standard synthesis and analysis transforms

from typing import List, Optional

import torch
import torch.nn as nn
from compressai.layers import GDN, AttentionBlock
from neural.layers_utils import make_conv, make_deconv
from torch import Tensor


class ConvGDNAnalysis(nn.Module):
    def __init__(
        self, network_channels: int = 128, compression_channels: int = 192
    ) -> None:
        """
        Analysis transfrom from scale hyperprior (https://arxiv.org/abs/1802.01436)

        Encodes each image in a video independently.
        """
        super().__init__()
        self._compression_channels = compression_channels

        self.transforms = nn.Sequential(
            make_conv(3, network_channels, kernel_size=5, stride=2),
            GDN(network_channels),
            make_conv(network_channels, network_channels, kernel_size=5, stride=2),
            GDN(network_channels),
            make_conv(network_channels, network_channels, kernel_size=5, stride=2),
            GDN(network_channels),
            make_conv(network_channels, compression_channels, kernel_size=5, stride=2),
        )
        self._num_down = 4

    @property
    def compression_channels(self) -> int:
        return self._compression_channels

    @property
    def num_downsampling_layers(self) -> int:
        return self._num_down

    def forward(self, video_frames: Tensor) -> Tensor:
        """
        Args:
            video_frames: frames of a batch of clips. Expected shape [B, T, C, H, W],
                which is reshaped to [BxT, C, H, W], hyperprior model encoder is applied
                and output is reshaped back to [B, T, <compression_channels>, h, w].
        Returns:
            embeddings: embeddings of shape [B, T, <compression_channels>, h, w], obtained
                by running ScaleHyperprior.image_analysis().
        """
        assert (
            video_frames.dim() == 5
        ), f"Expected [B, T, C, H, W] got {video_frames.shape}"
        embeddings = self.transforms(video_frames.reshape(-1, *video_frames.shape[2:]))
        return embeddings.reshape(*video_frames.shape[:2], *embeddings.shape[1:])


class ConvGDNSynthesis(nn.Module):
    def __init__(
        self, network_channels: int = 128, compression_channels: int = 192
    ) -> None:
        """
        Synthesis transfrom from scale hyperprior (https://arxiv.org/abs/1802.01436)

        Decodes each image in a video independently
        """

        super().__init__()
        self._compression_channels = 192

        self.transforms = nn.Sequential(
            make_deconv(
                compression_channels, network_channels, kernel_size=5, stride=2
            ),
            GDN(network_channels, inverse=True),
            make_deconv(network_channels, network_channels, kernel_size=5, stride=2),
            GDN(network_channels, inverse=True),
            make_deconv(network_channels, network_channels, kernel_size=5, stride=2),
            GDN(network_channels, inverse=True),
            make_deconv(network_channels, 3, kernel_size=5, stride=2),
        )

    @property
    def compression_channels(self) -> int:
        return self._compression_channels

    def forward(self, x: Tensor, frames_shape: torch.Size) -> Tensor:
        """
        Args:
            x: the (reconstructed) latent embdeddings to be decoded to images,
                expected shape [B, T, C, H, W]
            frames_shape: shape of the video clip to be reconstructed.
        Returns:
            reconstruction: reconstruction of the original video clip with shape
                [B, T, C, H, W] = frames_shape.
        """
        assert x.dim() == 5, f"Expected [B, T, C, H, W] got {x.shape}"
        # Treat T as part of the Batch dimension, storing values to reshape back
        B, T, *_ = x.shape
        x = x.reshape(-1, *x.shape[2:])
        assert len(frames_shape) == 5

        x = self.transforms(x)  # final reconstruction
        x = x.reshape(B, T, *x.shape[1:])
        return x[..., : frames_shape[-2], : frames_shape[-1]]


class ResidualUnit(nn.Module):
    """Simple residual unit"""

    def __init__(self, N: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            make_conv(N, N // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            make_conv(N // 2, N // 2, kernel_size=3),
            nn.ReLU(inplace=True),
            make_conv(N // 2, N, kernel_size=1),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv(x)
        out += identity
        out = self.activation(out)
        return out


class ELICAnalysis(nn.Module):
    def __init__(
        self,
        num_residual_blocks=3,
        channels: List[int] = [128, 160, 192, 192],
        compression_channels: Optional[int] = None,
        max_frames: Optional[int] = None,
    ) -> None:
        """Analysis transform from ELIC (https://arxiv.org/abs/2203.10886), which
        can be configured to match the one from "Devil's in the Details"
        (https://arxiv.org/abs/2203.08450).

        Args:
            num_residual_blocks: defaults to 3.
            channels: defaults to [128, 160, 192, 192].
            compression_channels: optional, defaults to None. If provided, it must equal
                the last element of `channels`.
            max_frames: optional, defaults to None. If provided, the input is chunked
                into max_frames elements, otherwise the entire batch is processed at
                once. This is useful when large sequences are to be processed and can
                be used to manage memory a bit better.
        """
        super().__init__()
        if len(channels) != 4:
            raise ValueError(f"ELIC uses 4 conv layers (not {len(channels)}).")
        if compression_channels is not None and compression_channels != channels[-1]:
            raise ValueError(
                "output_channels specified but does not match channels: "
                f"{compression_channels} vs. {channels}"
            )
        self._compression_channels = (
            compression_channels if compression_channels is not None else channels[-1]
        )
        self._max_frames = max_frames

        def res_units(N):
            return [ResidualUnit(N) for _ in range(num_residual_blocks)]

        channels = [3] + channels

        self.transforms = nn.Sequential(
            make_conv(channels[0], channels[1], kernel_size=5, stride=2),
            *res_units(channels[1]),
            make_conv(channels[1], channels[2], kernel_size=5, stride=2),
            *res_units(channels[2]),
            AttentionBlock(channels[2]),
            make_conv(channels[2], channels[3], kernel_size=5, stride=2),
            *res_units(channels[3]),
            make_conv(channels[3], channels[4], kernel_size=5, stride=2),
            AttentionBlock(channels[4]),
        )

    @property
    def compression_channels(self) -> int:
        return self._compression_channels

    def forward(self, x: Tensor) -> Tensor:
        assert x.dim() == 5, f"Expected [B, T, C, H, W] got {x.shape}"
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        if self._max_frames is not None and B * T > self._max_frames:
            assert (B * T) % self._max_frames == 0, "Can't reshape!"
            f = (B * T) // self._max_frames
            x = x.reshape(f, self._max_frames, *x.shape[1:])
            x = torch.stack([self.transforms(chunk) for chunk in x], dim=0)
            x = torch.flatten(x, start_dim=0, end_dim=1)
        else:
            x = self.transforms(x)
        return x.reshape(B, T, *x.shape[1:])


class ELICSynthesis(nn.Module):
    def __init__(
        self,
        num_residual_blocks=3,
        channels: List[int] = [192, 160, 128, 3],
        output_channels: Optional[int] = None,
        max_frames: Optional[int] = None,
    ) -> None:
        """
        Synthesis transform from ELIC (https://arxiv.org/abs/2203.10886).

        Args:
            num_residual_blocks: defaults to 3.
            channels: _defaults to [192, 160, 128, 3].
            output_channels: optional, defaults to None. If provided, it must equal
                the last element of `channels`.
            max_frames: optional, defaults to None. If provided, the input is chunked
                into max_frames elements, otherwise the entire batch is processed at
                once. This is useful when large sequences are to be processed and can
                be used to manage memory a bit better.
        """
        super().__init__()
        if len(channels) != 4:
            raise ValueError(f"ELIC uses 4 conv layers (not {channels}).")
        if output_channels is not None and output_channels != channels[-1]:
            raise ValueError(
                "output_channels specified but does not match channels: "
                f"{output_channels} vs. {channels}"
            )

        self._compression_channels = channels[0]
        self._max_frames = max_frames

        def res_units(N: int) -> List:
            return [ResidualUnit(N) for _ in range(num_residual_blocks)]

        channels = [channels[0]] + channels
        self.transforms = nn.Sequential(
            AttentionBlock(channels[0]),
            make_deconv(channels[0], out_channels=channels[1], kernel_size=5, stride=2),
            *res_units(channels[1]),
            make_deconv(channels[1], out_channels=channels[2], kernel_size=5, stride=2),
            AttentionBlock(channels[2]),
            *res_units(channels[2]),
            make_deconv(channels[2], out_channels=channels[3], kernel_size=5, stride=2),
            *res_units(channels[3]),
            make_deconv(channels[3], out_channels=channels[4], kernel_size=5, stride=2),
        )

    @property
    def compression_channels(self) -> int:
        return self._compression_channels

    def forward(self, x: Tensor, frames_shape: torch.Size) -> Tensor:
        """
        Args:
            x: the (reconstructed) latent embdeddings to be decoded to images.
            frames_shape: shape of the video clip to be reconstructed.
        Returns:
            reconstruction: reconstruction of the original video clip with shape
                [B, T, C, H, W] = frames_shape.
        """

        assert x.dim() == 5, f"Expected [B, T, C, H, W] got {x.shape}"
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        if self._max_frames is not None and B * T > self._max_frames:
            assert (B * T) % self._max_frames == 0, "Can't reshape!"
            f = (B * T) // self._max_frames
            x = x.reshape(f, self._max_frames, C, H, W)
            x = torch.stack([self.transforms(chunk) for chunk in x], dim=0)
            x = torch.flatten(x, start_dim=0, end_dim=1)
        else:
            x = self.transforms(x)

        x = x.reshape(B, T, *x.shape[1:])
        return x[..., : frames_shape[-2], : frames_shape[-1]]
