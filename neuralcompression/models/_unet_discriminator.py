# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from functools import partial
from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.parametrizations import spectral_norm

from ._discriminator import Discriminator


def _verify_norm_type(norm_type: str):
    if norm_type not in ("batch", "spectral", "instance", "identity"):
        raise ValueError(f"Unrecognized norm_type {norm_type}")


def apply_conv_norm(module: nn.Module, norm_type: str = "identity") -> nn.Module:
    _verify_norm_type(norm_type)
    if norm_type == "spectral":
        return spectral_norm(module)
    else:
        return module


# TODO: Make this a config?
def build_norm_layer(chans: int, norm_type: str) -> nn.Module:
    _verify_norm_type(norm_type)
    if norm_type == "batch":
        return nn.BatchNorm2d(chans)
    elif norm_type == "instance":
        return nn.InstanceNorm2d(chans)
    else:
        return nn.Identity()


def _conv(
    in_planes: int,
    out_planes: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    bias: bool = True,
):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )


class _UnetLevel(nn.Module):
    def __init__(
        self,
        in_planes: int,
        down_planes: int,
        child: nn.Module,
        out_planes: Optional[int] = None,
        top: bool = False,
        clip_upsample: bool = False,
        norm_type: str = "identity",
    ):
        super().__init__()
        self.in_planes = in_planes
        self.down_planes = down_planes
        if out_planes is None:
            out_planes = in_planes
        self.out_planes = out_planes
        self.child = child
        self.clip_upsample = clip_upsample
        self.norm_type = norm_type

        use_bias = True
        if norm_type == "batch":
            use_bias = False

        norm_layer = partial(build_norm_layer, norm_type=norm_type)
        conv_norm = partial(apply_conv_norm, norm_type=norm_type)

        self.left_block = nn.Sequential(
            nn.Identity() if top else norm_layer(in_planes),
            nn.Identity() if top else nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv_norm(
                _conv(in_planes=in_planes, out_planes=down_planes, bias=use_bias)
            ),
            norm_layer(down_planes),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv_norm(_conv(in_planes=down_planes, out_planes=down_planes)),
            nn.AvgPool2d(2),
        )
        if in_planes != down_planes:
            left_shortcut_conv = conv_norm(
                _conv(
                    in_planes=in_planes,
                    out_planes=down_planes,
                    kernel_size=1,
                    padding=0,
                    bias=True,
                )
            )
        else:
            left_shortcut_conv = nn.Identity()

        self.left_shortcut = nn.Sequential(
            left_shortcut_conv,
            nn.AvgPool2d(2),
        )

        self.right_block: nn.Module
        self.right_shortcut: Union[nn.Module, Callable]
        if clip_upsample is True:
            self.right_block = nn.Identity()
            self.right_shortcut = lambda x: torch.tensor(0.0)
        else:
            self.right_block = nn.Sequential(
                norm_layer(down_planes),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Upsample(scale_factor=2),
                conv_norm(
                    _conv(in_planes=down_planes, out_planes=out_planes, bias=use_bias)
                ),
                norm_layer(out_planes),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                conv_norm(_conv(in_planes=out_planes, out_planes=out_planes)),
            )
            if down_planes != out_planes:
                right_shortcut_conv = conv_norm(
                    _conv(
                        in_planes=down_planes,
                        out_planes=out_planes,
                        kernel_size=1,
                        padding=0,
                        bias=True,
                    )
                )
            else:
                right_shortcut_conv = nn.Identity()

            self.right_shortcut = nn.Sequential(
                right_shortcut_conv, nn.Upsample(scale_factor=2)
            )

    def forward(self, image: Tensor) -> Tensor:
        image = self.child(self.left_block(image) + self.left_shortcut(image))
        return self.right_block(image) + self.right_shortcut(image)


class UnetDiscriminator(Discriminator):
    """
    OASIS U-Net Discriminator

    This class implements the U-Net Discriminator as documented in the
    following paper:

    You Only Need Adversarial Supervision for Semantic Image Synthesis
    V Sushko, E Schönfeld, D Zhang, J Gall, B Schiele, A Khoreva

    It includes upsampling path modifications to calculate the discriminator
    used in neural compression paper:

    Improving Statistical Fidelity for Neural Image Compression with Implicit
    Local Likelihood Models
    MJ Muckley, A El-Nouby, K Ullrich, H Jégou, J Verbeek

    Unlike the model used in the paper, this model can be clipped in its
    upsampling path by setting ``output_downsampling_factor''. For example,
    if you would like to clip the upsampling path at a 16 x 16 resolution for
    a 256 x 256 image, you should set ``output_downsampling_factor'' to 16.

    Args:
        output_channels: Number of semantic classes or spatial label maps.
        output_downsampling_factor: The upsampling path should be clipped such
            that the output latent the original image resolution divided by
            this factor.
        chans: Channels for the U-Net. If none are passed, the default from the
            OASIS paper are used (``[3, 128, 128, 256, 256, 512, 512]'').
        norm_type: Type of normalization layer. One of ``('spectral',
            'instance', 'batch', 'identity')''.
    """

    def __init__(
        self,
        output_channels: int,
        output_downsampling_factor: int,
        chans: Optional[List[int]] = None,
        norm_type: str = "identity",
    ):
        super().__init__()

        # check if factor is power-of-2
        if not (
            (output_downsampling_factor & (output_downsampling_factor - 1) == 0)
            and output_downsampling_factor != 0
        ):
            raise ValueError("output_downsampling_factor must be power-of-2")
        num_downsamples = int(math.log2(output_downsampling_factor))

        # default chans: [3, 128, 128, 256, 256, 512, 512]
        if chans is None:
            self.channels = [3, 128, 128, 256, 256, 512, 512]
        else:
            self.channels = chans

        model: nn.Module = nn.Identity()
        lastchan = self.channels[-1]
        num_upsamples = len(self.channels) - 1 - num_downsamples
        clip_upsample = False
        for count, chan in enumerate(reversed(self.channels[1:-1])):
            if count == num_upsamples:
                clip_upsample = True

            model = _UnetLevel(
                in_planes=chan,
                out_planes=(64 if count == num_upsamples - 1 else None),
                down_planes=lastchan,
                child=model,
                clip_upsample=clip_upsample,
                norm_type=norm_type,
            )
            lastchan = chan

        model = _UnetLevel(
            in_planes=self.channels[0],
            down_planes=lastchan,
            out_planes=64,
            child=model,
            top=True,
            clip_upsample=clip_upsample,
            norm_type=norm_type,
        )

        self.model = nn.Sequential(model, nn.Conv2d(64, output_channels, 1, 1, 0))

    @property
    def is_conditional(self) -> bool:
        return False

    def forward(self, image: Tensor) -> Tensor:
        return self.model(image)


class ConditionalUnetDiscriminator(nn.Module):
    """
    OASIS U-Net Discriminator with Context

    This class implements the U-Net Discriminator as documented in the
    following paper:

    You Only Need Adversarial Supervision for Semantic Image Synthesis
    V Sushko, E Schönfeld, D Zhang, J Gall, B Schiele, A Khoreva

    It includes upsampling path modifications to calculate the discriminator
    used in neural compression paper:

    Improving Statistical Fidelity for Neural Image Compression with Implicit
    Local Likelihood Models
    MJ Muckley, A El-Nouby, K Ullrich, H Jégou, J Verbeek

    Unlike the model used in the paper, this model can be clipped in its
    upsampling path by setting ``output_downsampling_factor''. For example,
    if you would like to clip the upsampling path at a 16 x 16 resolution for
    a 256 x 256 image, you should set ``output_downsampling_factor'' to 16.

    In addition to the features of ``UnetDiscriminator'', this class also
    allows the passing of contxt information as documented in the HiFiC paper
    for PatchGAN.

    Args:
        output_channels: Number of semantic classes or spatial label maps.
        output_downsampling_factor: The upsampling path should be clipped such
            that the output latent the original image resolution divided by
            this factor.
        context_in: Number of channels for the context.
        context_out: Number of channels for the initial context processing to
            output.
        chans: Channels for the U-Net. If none are passed, the default from the
            OASIS paper are used (``[3, 128, 128, 256, 256, 512, 512]'').
        norm_type: Type of normalization layer. One of ``('spectral',
            'instance', 'batch', 'identity')''.
    """

    def __init__(
        self,
        output_channels: int,
        output_downsampling_factor: int,
        context_in: int,
        context_out: int = 16,
        chans: Optional[List[int]] = None,
        norm_type: str = "identity",
    ):
        super().__init__()
        if chans is None:
            unet_channels = [3 + context_out, 128, 128, 256, 256, 512, 512]
        else:
            unet_channels = chans
        self._unet = UnetDiscriminator(
            output_channels=output_channels,
            output_downsampling_factor=output_downsampling_factor,
            chans=unet_channels,
            norm_type=norm_type,
        )
        self.input_sequence = nn.Sequential(
            nn.Conv2d(
                context_in,
                context_out,
                kernel_size=3,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    @property
    def is_conditional(self) -> bool:
        return True

    def forward(self, image: Tensor, context: Tensor) -> Tensor:
        _, _, ny, nx = image.shape
        image = torch.cat(
            (
                image,
                F.interpolate(
                    self.input_sequence(context), size=(ny, nx), mode="nearest"
                ),
            ),
            dim=1,
        )

        return self._unet(image)
