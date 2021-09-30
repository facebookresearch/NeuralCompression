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

        self.input_sequence = torch.nn.Sequential(
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(
                self.features,
                12,
                (3, 3),
                padding=1,
                padding_mode="reflect",
            ),
            torch.nn.Upsample(None, 16, "nearest"),
        )

        convolutions = []

        in_channels = self.image_dimensions[0] + 12

        for out_channels in (64, 128, 256, 512):
            convolution = torch.nn.Conv2d(
                in_channels,
                out_channels,
                (4, 4),
                (2, 2),
                padding=1,
                padding_mode="reflect",
            )

            convolutions += [torch.nn.utils.spectral_norm(convolution)]

            in_channels = out_channels

        convolutions += [torch.nn.Conv2d(512, 1, (1, 1), (1, 1))]

        self.convolutions = torch.nn.Sequential(*convolutions)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        predictions = self.convolutions(
            torch.cat((x, self.input_sequence(y)), 1)
        ).view(-1, 1)

        return torch.sigmoid(predictions), predictions
