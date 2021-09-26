import collections
import enum
import functools

import numpy
import torch
import torch.nn

from ._hific_discriminator import _HiFiCDiscriminator
from ._hific_encoder import _HiFiCEncoder
from ._hific_generator import _HiFiCGenerator
from ...functional import generative_loss, weighted_rate_loss

DiscriminatorOutput = collections.namedtuple(
    "DiscriminatorOutput",
    [
        "discriminator_authentic",
        "discriminator_synthetic",
        "discriminator_authentic_predictions",
        "discriminator_synthetic_predictions"
    ]
)

IntermediateData = collections.namedtuple(
    "IntermediateData",
    [
        "input_image",
        "reconstruction",
        "quantized",
        "nbpp",
        "qbpp",
    ]
)


class ModelMode(enum.Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    EVALUATION = "evaluation"


class ModelType(enum.Enum):
    COMPRESSION = "compression"
    COMPRESSION_GAN = "compression_gan"


class HiFiC(torch.nn.Module):
    def __init__(self):
        super(HiFiC, self).__init__()

    def discriminator_forward(self, intermediate_data: IntermediateData, generate: bool) -> DiscriminatorOutput:
        generated_authentic = intermediate_data.input_image
        generated_synthetic = intermediate_data.reconstruction

        if not generate:
            generated_synthetic = generated_synthetic.detach()

        generated = torch.cat([generated_authentic, generated_synthetic], dim=0)

        quantized = torch.repeat_interleave(intermediate_data.quantized.detach(), repeats=2, dim=0)

        discriminated_images, discriminated_predictions = self.discriminator(
            generated,
            quantized,
        )

        discriminated_images = torch.squeeze(discriminated_images)

        discriminated_predictions = torch.squeeze(discriminated_predictions)

        discriminated_authentic, discriminated_synthetic = torch.chunk(
            discriminated_images, chunks=2, dim=0
        )

        discriminated_authentic_predictions, discriminated_synthetic_predictions = torch.chunk(
            discriminated_predictions, chunks=2, dim=0
        )

        return DiscriminatorOutput(
            discriminated_authentic,
            discriminated_synthetic,
            discriminated_authentic_predictions,
            discriminated_synthetic_predictions,
        )

    def distortion_loss(
            self,
            synthetic: torch.Tensor,
            authentic: torch.Tensor
    ) -> torch.Tensor:
        squared_difference = self.squared_difference(synthetic * 255.0, authentic * 255.0)

        return torch.mean(squared_difference)

    def perceptual_loss_wrapper(
            self,
            synthetic: torch.Tensor,
            authentic: torch.Tensor,
            normalize: bool = True
    ) -> torch.Tensor:
        perceptual_loss = self.perceptual_loss.forward(synthetic, authentic, normalize=normalize)

        return torch.mean(perceptual_loss)
