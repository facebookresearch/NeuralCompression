import typing

import torch
import torch.nn
import torch.nn.functional


class _HiFiCDiscriminator(torch.nn.Module):
    def __init__(
            self,
            image_dimensions: typing.Tuple[int] = (3, 256, 256),
            feature_dimensions: typing.Tuple[int] = (220, 16, 16),
            features: int = 220,
            spectral_norm=True
    ):
        super(_HiFiCDiscriminator, self).__init__()

        self.image_dimensions = image_dimensions

        self.feature_dimensions = feature_dimensions

        self.features = features

        self.activation = torch.nn.LeakyReLU(0.2)

        self.extract_features = torch.nn.Conv2d(self.features, 12, 3, padding=1, padding_mode="reflect")

        self.upsample = torch.nn.Upsample(None, 16, "nearest")

        if spectral_norm is True:
            norm = torch.nn.utils.spectral_norm
        else:
            norm = torch.nn.utils.weight_norm

        filters = (64, 128, 256, 512)

        kwargs = {"padding": 1, "padding_mode": "reflect", "stride": 2}

        self.conv1 = norm(torch.nn.Conv2d(self.image_dimensions[0] + 12, filters[0], 4, **kwargs))

        self.conv2 = norm(torch.nn.Conv2d(filters[0], filters[1], 4, **kwargs))
        self.conv3 = norm(torch.nn.Conv2d(filters[1], filters[2], 4, **kwargs))
        self.conv4 = norm(torch.nn.Conv2d(filters[2], filters[3], 4, **kwargs))

        self.discriminate = torch.nn.Conv2d(filters[3], 1, 1, 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        y = self.activation(self.extract_features(y))

        y = self.upsample(y)

        x = torch.cat((x, y), 1)

        x = self.activation(self.conv1(x))

        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        features = self.discriminate(x).view(-1, 1)

        return torch.sigmoid(features), features
