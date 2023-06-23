# Copyright (c) 2020 Patrick Esser and Robin Rombach and Björn Ommer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE.
#
# Modifications
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from neuralcompression.layers import XCA, ChannelNorm2D

from .._outputs._vqvae_autoencoder_output import VqVaeAutoencoderOutput


def _nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


class Normalize(torch.nn.GroupNorm):
    def __init__(self, in_channels: int):
        super().__init__(num_groups=16, num_channels=in_channels, eps=1e-6, affine=True)


def create_norm(in_channels: int, norm_type: str):
    if norm_type == "group":
        return Normalize(in_channels)
    elif norm_type == "channel":
        return ChannelNorm2D(in_channels)
    else:
        raise ValueError(f"Unrecognized norm_type {norm_type}")


class _Upsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class _Downsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x: Tensor) -> Tensor:
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class _ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        norm_type: str = "group",
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = create_norm(in_channels, norm_type)
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = create_norm(out_channels, norm_type)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x: Tensor) -> Tensor:
        h = x
        h = self.norm1(h)
        h = _nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = _nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class _AttnBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        resolution_level: Optional[int] = None,
        norm_type: str = "group",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.resolution_level = resolution_level

        self.norm = create_norm(in_channels, norm_type=norm_type)
        self.xca = XCA(dim=in_channels, num_heads=1, qkv_bias=True)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, chans, num_y, num_x = x.shape
        h_ = self.norm(x).view(batch_size, chans, -1).permute(0, 2, 1)
        h_ = self.xca(h_)
        h_ = h_.permute(0, 2, 1).reshape(batch_size, chans, num_y, num_x)

        return x + h_


class _VqVaeXCiTConvEncoder(nn.Module):
    def __init__(
        self,
        ch: int = 128,
        ch_mult: Optional[Sequence] = None,
        num_res_blocks: int = 2,
        attn_resolutions: Optional[Sequence] = None,
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int = 3,
        resolution: int = 256,
        embed_dim: int = 256,
        norm_type: str = "group",
    ):
        super().__init__()
        if ch_mult is None:
            ch_mult = (1, 1, 2, 2, 4)
        if attn_resolutions is None:
            attn_resolutions = [16]

        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.grid_size = attn_resolutions

        if norm_type == "channel_last":
            norm_type = "group"

        # downsampling
        self.conv_in = torch.nn.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(
                    _ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                        norm_type=norm_type,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(_AttnBlock(block_in, curr_res, norm_type=norm_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = _Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Sequential(
            _ResnetBlock(
                in_channels=block_in,
                out_channels=block_in,
                dropout=dropout,
                norm_type=norm_type,
            ),
            _AttnBlock(block_in, curr_res, norm_type=norm_type),
            _ResnetBlock(
                in_channels=block_in,
                out_channels=block_in,
                dropout=dropout,
                norm_type=norm_type,
            ),
        )

        # end
        self.norm_out = create_norm(block_in, norm_type=norm_type)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            embed_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x: Tensor) -> Tensor:
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid(h)

        # end
        h = self.norm_out(h)
        h = _nonlinearity(h)
        h = self.conv_out(h)

        return h


class _VqvaeXCiTConvDecoder(nn.Module):
    def __init__(
        self,
        ch: int = 128,
        out_ch: int = 3,
        ch_mult: Optional[Sequence] = None,
        num_res_blocks: int = 2,
        attn_resolutions: Optional[Sequence] = None,
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int = 3,
        resolution: int = 256,
        embed_dim: int = 256,
        norm_type: str = "group",
    ):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        if ch_mult is None:
            ch_mult = (1, 1, 2, 2, 4)
        if attn_resolutions is None:
            attn_resolutions = [16]

        if norm_type == "channel_last":
            last_norm = "channel"
            norm_type = "group"
        else:
            last_norm = norm_type

        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, embed_dim, curr_res, curr_res)
        self.logger.info(
            "Working with z of shape {} = {} dimensions.".format(
                self.z_shape, np.prod(self.z_shape)
            )
        )

        # z to block_in
        self.conv_in = torch.nn.Conv2d(
            embed_dim, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Sequential(
            _ResnetBlock(
                in_channels=block_in,
                out_channels=block_in,
                dropout=dropout,
                norm_type=norm_type,
            ),
            _AttnBlock(block_in, curr_res, norm_type=norm_type),
            _ResnetBlock(
                in_channels=block_in,
                out_channels=block_in,
                dropout=dropout,
                norm_type=norm_type,
            ),
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(
                    _ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                        norm_type=norm_type,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(_AttnBlock(block_in, curr_res, norm_type=norm_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = _Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = create_norm(block_in, norm_type=last_norm)
        self.conv_out = torch.nn.Conv2d(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )

    def forward(self, z: Tensor) -> Tensor:
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = _nonlinearity(h)
        h = self.conv_out(h)

        return h


class VqVaeXCiTAutoencoder(nn.Module):
    """
    VQ-VAE model with XCiT attention.

    This model follows the overall architecture of the autoencoder from the
    VQ-GAN paper:

    Taming Transformers for High-Resolution Image Synthesis
    Patrick Esser, Robin Rombach, Bjorn Ommer

    However, there is one key modification: the use of XCiT attention as
    opposed to standard attention. This allows the resulting autoencoders to
    easily handle very high-resolution images that are used in compression.

    XCiT attention was introduced in the following paper:

    XCiT: Cross-Covariance Image Transformers
    A El-Nouby, H Touvron, M Caron, P Bojanowski, M Douze, A Joulin, I Laptev,
    N Neverova, G Synnaeve, J Verbeek, H Jegou

    This class does not implement the full VQ-VAE, just the encoder/decoder.
    The user is expected to pass in their own bottleneck and quantization
    operations via the `bottleneck_op` functional parameter.

    Args:
        ch: Number of top-level convolution channels.
        out_ch: Number of output channels.
        ch_mult: A multiplication factor for scaling the number of channels.
        num_res_blocks: Number of residual blocks for convolution layers.
        attn_resolution: The resolutions at which to apply attention.
        dropout: Dropout probability.
        resamp_with_conv: Whether to resample in upsampling layers with
            convolutions.
        in_channels: Number of input channels.
        resolution: Overall image resolution. Used for estimating size of
            latent on model instantiation.
        embed_dim: Size of embedding dimension.
        freeze_encoder: Whether to freeze the encoder.
        freeze_bottleneck: Whether to freeze the bottleneck (codebook).
        norm_type: Type of normalization.
    """

    def __init__(
        self,
        ch: int = 128,
        out_ch: int = 3,
        ch_mult: Optional[Sequence] = None,
        num_res_blocks: int = 2,
        attn_resolutions: Optional[Sequence] = None,
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int = 3,
        resolution: int = 256,
        embed_dim: int = 256,
        bottleneck_op: Optional[nn.Module] = None,
        freeze_encoder: bool = False,
        freeze_bottleneck: bool = False,
        norm_type: str = "group",
    ):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        if ch_mult is None:
            ch_mult = (1, 1, 2, 2, 4)
        if attn_resolutions is None:
            attn_resolutions = [16]

        self.freeze_encoder = freeze_encoder
        self.freeze_bottleneck = freeze_bottleneck
        self.factor = 2 ** (len(ch_mult) - 1)

        self.encoder = _VqVaeXCiTConvEncoder(
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            resamp_with_conv=resamp_with_conv,
            in_channels=in_channels,
            resolution=resolution,
            embed_dim=embed_dim,
            norm_type=norm_type,
        )
        self.decoder = _VqvaeXCiTConvDecoder(
            ch=ch,
            out_ch=out_ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            resamp_with_conv=resamp_with_conv,
            in_channels=in_channels,
            resolution=resolution,
            embed_dim=embed_dim,
            norm_type=norm_type,
        )
        self.bottleneck_op = bottleneck_op

        if self.freeze_encoder:
            self.logger.info("Freezing encoder!")
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad_(False)

        if self.freeze_bottleneck:
            self.logger.info("Freezing bottleneck!")
            if self.bottleneck_op is not None:
                self.bottleneck_op.eval()
                for param in self.bottleneck_op.parameters():
                    param.requires_grad_(False)

    def _run_encoder(self, image: Tensor) -> VqVaeAutoencoderOutput:
        return VqVaeAutoencoderOutput(latent=self.encoder(image))

    def forward(self, image: Tensor) -> VqVaeAutoencoderOutput:
        if self.freeze_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad_(False)

        if self.freeze_bottleneck:
            if self.bottleneck_op is not None:
                self.bottleneck_op.eval()
                for param in self.bottleneck_op.parameters():
                    param.requires_grad_(False)

        # pad image if it's not divisible by downsamples
        _, _, height, width = image.shape
        if not self.training:
            pad_height = (self.factor - (height - self.factor)) % self.factor
            pad_width = (self.factor - (width - self.factor)) % self.factor
            if pad_height != 0 or pad_width != 0:
                image = F.pad(image, (0, pad_width, 0, pad_height), mode="reflect")

        output = self._run_encoder(image)

        if self.bottleneck_op is not None:
            output = self.bottleneck_op(output)

        output.image = self.decoder(output.latent)[:, :, :height, :width]

        return output
