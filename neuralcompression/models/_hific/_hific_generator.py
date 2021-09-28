import typing

import torch
import torch.nn
import torch.nn.functional

from ._channel_norm_2d import _channel_norm_2d


class _ResidualBlock(torch.nn.Module):
    def __init__(
            self,
            input_dimensions,
            kernel_size=(3, 3),
            stride=(1, 1),
    ):
        super(_ResidualBlock, self).__init__()

        self.activation = torch.nn.ReLU()

        self.pad = torch.nn.ReflectionPad2d(int((kernel_size[0] - 1) / 2))

        self.conv_a = torch.nn.Conv2d(input_dimensions[1], input_dimensions[1], kernel_size, stride)
        self.conv_b = torch.nn.Conv2d(input_dimensions[1], input_dimensions[1], kernel_size, stride)

        kwargs = {
            "affine": True,
            "momentum": 0.1,
            "track_running_stats": False,
        }

        self.norm_a = _channel_norm_2d(input_dimensions[1], **kwargs)
        self.norm_b = _channel_norm_2d(input_dimensions[1], **kwargs)

    def forward(self, x):
        identity_map = x

        features = self.pad(x)

        features = self.conv_a(features)
        features = self.norm_a(features)

        features = self.activation(features)

        features = self.pad(features)

        features = self.conv_b(features)
        features = self.norm_b(features)

        return torch.add(features, identity_map)


class HiFiCGenerator(torch.nn.Module):
    def __init__(
            self,
            input_dimensions: typing.Tuple[int, int, int] = (3, 256, 256),
            batch_size: int = 8,
            latent_features: int = 220,
            n_residual_blocks: int = 9,
    ):
        super(HiFiCGenerator, self).__init__()

        self.n_residual_blocks = n_residual_blocks

        filters = [960, 480, 240, 120, 60]

        conv_kwargs = {
            "output_padding": 1,
            "padding": 1,
            "stride": 2,
        }

        norm_kwargs = {
            "affine": True,
            "momentum": 0.1,
            "track_running_stats": False,
        }

        self.block_0 = torch.nn.Sequential(
            _channel_norm_2d(latent_features, **norm_kwargs),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(latent_features, filters[0], (3, 3), (1, 1)),
            _channel_norm_2d(filters[0], **norm_kwargs),
        )

        for m in range(self.n_residual_blocks):
            residual_block_m = _ResidualBlock((batch_size, filters[0], *input_dimensions[1:]))

            self.add_module(f"_ResidualBlock_{str(m)}", residual_block_m)

        self.block_1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(filters[0], filters[1], (3, 3), **conv_kwargs),
            _channel_norm_2d(filters[1], **norm_kwargs),
            torch.nn.ReLU(),
        )

        self.block_2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(filters[1], filters[2], (3, 3), **conv_kwargs),
            _channel_norm_2d(filters[2], **norm_kwargs),
            torch.nn.ReLU(),
        )

        self.block_3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(filters[2], filters[3], (3, 3), **conv_kwargs),
            _channel_norm_2d(filters[3], **norm_kwargs),
            torch.nn.ReLU(),
        )

        self.block_4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(filters[3], filters[4], (3, 3), **conv_kwargs),
            _channel_norm_2d(filters[4], **norm_kwargs),
            torch.nn.ReLU(),
        )

        self.features = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(filters[-1], 3, (7, 7), (1, 1)),
        )

    def forward(self, x):
        block_0 = self.block_0(x)

        for m in range(self.n_residual_blocks):
            residual_block_m = getattr(self, f"_ResidualBlock_{str(m)}")

            if m == 0:
                x = residual_block_m(block_0)
            else:
                x = residual_block_m(x)

        x += block_0

        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)

        return self.features(x)
