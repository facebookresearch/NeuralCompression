import typing

import torch
import torch.nn
import torch.nn.functional


class HiFiCDiscriminator(torch.nn.Module):
    def __init__(
            self,
            image_dimensions: typing.Tuple[int, int, int] = (3, 256, 256),
            latent_features: int = 220,
    ):
        super(HiFiCDiscriminator, self).__init__()

        self.image_dimensions = image_dimensions

        self.features = latent_features

        self.activation = torch.nn.LeakyReLU(0.2)

        self.conv_0 = torch.nn.Conv2d(self.features, 12, (3, 3), padding=1, padding_mode="reflect")
        self.conv_0 = torch.nn.utils.spectral_norm(self.conv_0)

        self.upsample = torch.nn.Upsample(None, 16, "nearest")

        filters = (64, 128, 256, 512)

        kwargs = {
            "padding": 1,
            "padding_mode": "reflect",
            "stride": 2,
        }

        self.conv_1 = torch.nn.Conv2d(self.image_dimensions[0] + 12, filters[0], (4, 4), **kwargs)
        self.conv_1 = torch.nn.utils.spectral_norm(self.conv_1)

        self.conv_2 = torch.nn.Conv2d(filters[0], filters[1], (4, 4), **kwargs)
        self.conv_2 = torch.nn.utils.spectral_norm(self.conv_2)

        self.conv_3 = torch.nn.Conv2d(filters[1], filters[2], (4, 4), **kwargs)
        self.conv_3 = torch.nn.utils.spectral_norm(self.conv_3)

        self.conv_4 = torch.nn.Conv2d(filters[2], filters[3], (4, 4), **kwargs)
        self.conv_4 = torch.nn.utils.spectral_norm(self.conv_4)

        self.predictions = torch.nn.Conv2d(filters[3], 1, (1, 1), (1, 1))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        y = self.activation(self.conv_0(y))

        y = self.upsample(y)

        x = torch.cat((x, y), 1)

        x = self.activation(self.conv_1(x))

        x = self.activation(self.conv_2(x))
        x = self.activation(self.conv_3(x))
        x = self.activation(self.conv_4(x))

        predictions = self.predictions(x).view(-1, 1)

        return torch.sigmoid(predictions), predictions
