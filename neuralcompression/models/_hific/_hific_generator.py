import torch
import torch.nn
import torch.nn.functional

from _channel_norm_2d import _channel_norm_2d
from _instance_norm_2d import _instance_norm_2d
from _residual_block import _ResidualBlock


class _HiFiCGenerator(torch.nn.Module):
    def __init__(
            self,
            input_dimensions,
            batch_size,
            activation='relu',
            channel_norm=True,
            channels=16,
            n_residual_blocks=8,
            noise_dim=32,
            sample_noise=False,
    ):
        super(_HiFiCGenerator, self).__init__()

        self.n_residual_blocks = n_residual_blocks

        self.noise_dimension = noise_dim

        self.sample_noise = sample_noise

        activations = {
            "elu": "ELU",
            "leaky_relu": "LeakyReLU",
            "relu": "ReLU",
        }

        self.activation = getattr(torch.nn, activations[activation])

        self.n_upsampling_layers = 4

        if channel_norm is True:
            self.norm = _channel_norm_2d
        else:
            self.norm = _instance_norm_2d

        height, width = input_dimensions[1:]

        filters = [960, 480, 240, 120, 60]

        conv_kwargs = {"stride": 2, "padding": 1, "output_padding": 1}

        norm_kwargs = {"momentum": 0.1, "affine": True, "track_running_stats": False}

        self.block_0 = torch.nn.Sequential(
            self.norm(channels, **norm_kwargs),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(channels, filters[0], (3, 3), 1),
            self.norm(filters[0], **norm_kwargs),
        )

        if sample_noise is True:
            filters[0] += self.noise_dimension

        for m in range(self.n_residual_blocks):
            residual_block_m = _ResidualBlock(
                input_dims=(batch_size, filters[0], height, width),
                channel_norm=channel_norm,
                activation=activation
            )

            self.add_module(f"_ResidualBlock_{str(m)}", residual_block_m)

        self.block_1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(filters[0], filters[1], 3, **conv_kwargs),
            self.norm(filters[1], **norm_kwargs),
            self.activation(),
        )

        self.block_2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(filters[1], filters[2], 3, **conv_kwargs),
            self.norm(filters[2], **norm_kwargs),
            self.activation(),
        )

        self.block_3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(filters[2], filters[3], 3, **conv_kwargs),
            self.norm(filters[3], **norm_kwargs),
            self.activation(),
        )

        self.block_4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(filters[3], filters[4], 3, **conv_kwargs),
            self.norm(filters[4], **norm_kwargs),
            self.activation(),
        )

        self.generate = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(filters[-1], 3, (7, 7), 1),
        )

    def forward(self, x):
        block_0 = self.block_0(x)

        if self.sample_noise is True:
            b, c, h, w = tuple(block_0.size())

            sample = torch.randn((b, self.noise_dimension, h, w)).to(block_0)

            block_0 = torch.cat((block_0, sample), dim=1)

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

        return self.generate(x)
