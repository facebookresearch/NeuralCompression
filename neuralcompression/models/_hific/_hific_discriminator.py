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
                self.features, 12, (3, 3), padding=1, padding_mode="reflect"
            ),
            torch.nn.Upsample(None, 16, "nearest"),
        )

        filters = (64, 128, 256, 512)

        conv_sequence = []
        old_size = self.image_dimensions[0] + 12
        for current_filter in filters:
            conv_sequence.append(
                torch.nn.utils.spectral_norm(
                    torch.nn.Conv2d(
                        old_size,
                        current_filter,
                        (4, 4),
                        padding=1,
                        padding_mode="reflect",
                        stride=2,
                    )
                )
            )
            old_size = current_filter

        conv_sequence.append(torch.nn.Conv2d(filters[3], 1, (1, 1), (1, 1)))

        self.conv_sequence = torch.nn.Sequential(*conv_sequence)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        predictions = self.conv_sequence(
            torch.cat((x, self.input_sequence(y)), 1)
        ).view(-1, 1)

        return torch.sigmoid(predictions), predictions
