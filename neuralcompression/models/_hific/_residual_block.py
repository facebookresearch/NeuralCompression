import torch
import torch.nn
import torch.nn.functional

from ._channel_norm_2d import _channel_norm_2d
from ._instance_norm_2d import _instance_norm_2d


class _ResidualBlock(torch.nn.Module):
    def __init__(
            self,
            input_dims,
            kernel_size=3,
            stride=1,
            channel_norm=True,
            activation="relu"
    ):
        super(_ResidualBlock, self).__init__()

        self.activation = getattr(torch.nn.functional, activation)

        if channel_norm is True:
            self.interlayer_norm = _channel_norm_2d
        else:
            self.interlayer_norm = _instance_norm_2d

        self.pad = torch.nn.ReflectionPad2d(int((kernel_size - 1) / 2))

        self.conv_a = torch.nn.Conv2d(input_dims[1], input_dims[1], kernel_size, stride)
        self.conv_b = torch.nn.Conv2d(input_dims[1], input_dims[1], kernel_size, stride)

        kwargs = {"affine": True, "momentum": 0.1, "track_running_stats": False}

        self.norm_a = self.interlayer_norm(input_dims[1], **kwargs)
        self.norm_b = self.interlayer_norm(input_dims[1], **kwargs)

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
