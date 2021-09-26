import torch.nn

from ._channel_norm_2d import _channel_norm_2d
from ._instance_norm_2d import _instance_norm_2d


class Encoder(torch.nn.Module):
    def __init__(
            self,
            input_dimensions,
            activation="relu",
            channel_norm=True,
            channels=220,
    ):
        super(Encoder, self).__init__()

        activations = {
            "elu": "ELU",
            "leaky_relu": "LeakyReLU",
            "relu": "ReLU",
        }

        self.activation = getattr(torch.nn, activations[activation])

        self.n_downsampling_layers = 4

        if channel_norm:
            self.norm = _channel_norm_2d
        else:
            self.norm = _instance_norm_2d

        filters = (60, 120, 240, 480, 960)

        conv_kwargs = {
            "padding": 0,
            "padding_mode": "reflect",
            "stride": 2,
        }

        norm_kwargs = {
            "affine": True,
            "momentum": 0.1,
            "track_running_stats": False,
        }

        self.block_0 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(input_dimensions[0], filters[0], (7, 7), 1),
            self.norm(filters[0], **norm_kwargs),
            self.activation(),
        )

        self.block_1 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d((0, 1, 1, 0)),
            torch.nn.Conv2d(filters[0], filters[1], 3, **conv_kwargs),
            self.norm(filters[1], **norm_kwargs),
            self.activation(),
        )

        self.block_2 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d((0, 1, 1, 0)),
            torch.nn.Conv2d(filters[1], filters[2], 3, **conv_kwargs),
            self.norm(filters[2], **norm_kwargs),
            self.activation(),
        )

        self.block_3 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d((0, 1, 1, 0)),
            torch.nn.Conv2d(filters[2], filters[3], 3, **conv_kwargs),
            self.norm(filters[3], **norm_kwargs),
            self.activation(),
        )

        self.block_4 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d((0, 1, 1, 0)),
            torch.nn.Conv2d(filters[3], filters[4], 3, **conv_kwargs),
            self.norm(filters[4], **norm_kwargs),
            self.activation(),
        )

        self.encode = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(filters[4], channels, 3, 1),
        )

    def forward(self, x):
        x = self.block_0(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)

        return self.encode(x)
