import collections
import enum
import functools
import typing

import numpy
import torch
import torch.nn

from ._hific_discriminator import _HiFiCDiscriminator
from ._hific_encoder import _HiFiCEncoder
from ._hific_generator import _HiFiCGenerator
from .. import ScaleHyperprior
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
    discriminate: bool = False
    discriminator: typing.Optional[_HiFiCDiscriminator] = None
    discriminator_steps: int = 0
    normalize_input_image: bool = False
    likelihood_type: str = "gaussian"
    beta: float = 0.15
    step_counter: int = 0

    def __init__(
            self,
            args,
            image_dimensions: typing.Tuple[int] = (3, 256, 256),
            batch_size: int = 8,
            model_mode=ModelMode.TRAINING,
            model_type=ModelType.COMPRESSION
    ):
        super(HiFiC, self).__init__()

        self.args = args
        self.model_mode = model_mode
        self.model_type = model_type

        self.image_dimensions = image_dimensions
        self.batch_size = batch_size

        self.entropy_code = False

        if model_mode == ModelMode.EVALUATION:
            self.entropy_code = True

        self.encoder = _HiFiCEncoder(self.image_dimensions)

        self.generator = _HiFiCGenerator(self.image_dimensions, self.batch_size)

        self.prior = ScaleHyperprior()

        if self.model_type == ModelType.COMPRESSION_GAN and (self.model_mode != ModelMode.EVALUATION):
            self.discriminate = True

        if self.discriminate:
            self.discriminator_steps = self.discriminator_steps

            self.discriminator = _HiFiCDiscriminator()

            self.gan_loss = functools.partial(generative_loss, args.gan_loss_type)
        else:
            self.discriminator_steps = 0

            self.discriminator = None

        self.squared_difference = torch.nn.MSELoss(reduction="none")

        self.perceptual_loss = PerceptualLoss(
            model="net-lin",
            net="alex",
            use_gpu=torch.cuda.is_available(),
            gpu_ids=[args.gpu]
        )

    def store_loss(self, key, loss):
        assert type(loss) == float, "Call .item() on loss before storage"

        if self.training is True:
            storage = self.storage_train
        else:
            storage = self.storage_test

        if self.writeout is True:
            storage[key].append(loss)

    def compression_forward(self, x: torch.Tensor) -> IntermediateData:
        image_dims = tuple(x.size()[1:])  # (C,H,W)

        if self.model_mode == ModelMode.EVALUATION and (self.training is False):
            n_encoder_downsamples = self.encoder.n_downsampling_layers

            factor = 2 ** n_encoder_downsamples

            x = self.pad_factor(x, x.size()[2:], factor)

        # Encoder forward pass
        y = self.encoder(x)

        if self.model_mode == ModelMode.EVALUATION and (self.training is False):
            n_hyperencoder_downsamples = self.prior.analysis_net.n_downsampling_layers

            factor = 2 ** n_hyperencoder_downsamples

            y = self.pad_factor(y, y.size()[2:], factor)

        hyperinfo = self.prior(y, spatial_shape=x.size()[2:])

        latents_quantized = hyperinfo.decoded

        total_nbpp = hyperinfo.total_nbpp
        total_qbpp = hyperinfo.total_qbpp

        # Use quantized latents as input to G
        reconstruction = self.generator(latents_quantized)

        if self.normalize_input_image is True:
            reconstruction = torch.tanh(reconstruction)

        # Undo padding
        if self.model_mode == ModelMode.EVALUATION and (self.training is False):
            reconstruction = reconstruction[:, :, :image_dims[1], :image_dims[2]]

        intermediate_data = IntermediateData(x, reconstruction, latents_quantized, total_nbpp, total_qbpp)

        return intermediate_data, hyperinfo

    def discriminator_forward(
            self,
            intermediate_data: IntermediateData,
            generate: bool
    ) -> DiscriminatorOutput:
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

    def compression_loss(self, intermediate_data: IntermediateData, hyperinfo) -> torch.Tensor:
        x_real = intermediate_data.input_image
        x_gen = intermediate_data.reconstruction

        if self.normalize_input_image is True:
            x_real = (x_real + 1.) / 2.
            x_gen = (x_gen + 1.) / 2.

        distortion_loss = self.distortion_loss(x_gen, x_real)
        perceptual_loss = self.perceptual_loss_wrapper(x_gen, x_real, normalize=True)

        weighted_distortion = self.k_M * distortion_loss
        weighted_perceptual = self.k_P * perceptual_loss

        weighted_rate, rate_penalty = weighted_rate_loss(
            self.args,
            total_nbpp=intermediate_data.nbpp,
            total_qbpp=intermediate_data.qbpp,
            step_counter=self.step_counter,
            ignore_schedule=self.ignore_schedule
        )

        weighted_r_d_loss = weighted_rate + weighted_distortion

        weighted_compression_loss = weighted_r_d_loss + weighted_perceptual

        # Bookkeeping
        if self.step_counter % self.log_interval == 1:
            self.store_loss("rate_penalty", rate_penalty)
            self.store_loss("distortion", distortion_loss.item())
            self.store_loss("perceptual", perceptual_loss.item())
            self.store_loss("n_rate", intermediate_data.nbpp.item())
            self.store_loss("q_rate", intermediate_data.qbpp.item())
            self.store_loss("n_rate_latent", hyperinfo.latent_nbpp.item())
            self.store_loss("q_rate_latent", hyperinfo.latent_qbpp.item())
            self.store_loss("n_rate_hyperlatent", hyperinfo.hyperlatent_nbpp.item())
            self.store_loss("q_rate_hyperlatent", hyperinfo.hyperlatent_qbpp.item())

            self.store_loss("weighted_rate", weighted_rate.item())
            self.store_loss("weighted_distortion", weighted_distortion.item())
            self.store_loss("weighted_perceptual", weighted_perceptual.item())
            self.store_loss("weighted_R_D", weighted_r_d_loss.item())
            self.store_loss("weighted_compression_loss_sans_G", weighted_compression_loss.item())

        return weighted_compression_loss

    def generative_loss(
            self,
            intermediate_data: IntermediateData,
            generate: bool = False
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        discriminator_output = self.discriminator_forward(intermediate_data, generate)

        discriminator_loss = self.gan_loss(discriminator_output, mode="discriminator_loss")

        generator_loss = self.gan_loss(discriminator_output, mode="generator_loss")

        if self.step_counter % self.log_interval == 1:
            self.store_loss(
                "discriminator_authentic",
                torch.mean(discriminator_output.discriminator_authentic).item()
            )

            self.store_loss(
                "discriminator_synthetic",
                torch.mean(discriminator_output.discriminator_synthetic).item()
            )

            self.store_loss(
                "discriminator_loss",
                discriminator_loss.item()
            )

            self.store_loss(
                "generator_loss",
                generator_loss.item()
            )

            self.store_loss(
                "weighted_generative_loss",
                (self.beta * generator_loss).item()
            )

        return discriminator_loss, generator_loss

    def compress(self, x: torch.Tensor, silent=False) -> torch.Tensor:
        assert self.model_mode == ModelMode.EVALUATION and (self.training is False), (
            f"Set model mode to {ModelMode.EVALUATION} for compression.")

        spatial_shape = tuple(x.size()[2:])

        if self.model_mode == ModelMode.EVALUATION and (self.training is False):
            n_encoder_downsamples = self.encoder.n_downsampling_layers

            factor = 2 ** n_encoder_downsamples

            x = self.pad_factor(x, x.size()[2:], factor)

        # Encoder forward pass
        y = self.encoder(x)

        if self.model_mode == ModelMode.EVALUATION and (self.training is False):
            n_hyperencoder_downsamples = self.prior.analysis_net.n_downsampling_layers

            factor = 2 ** n_hyperencoder_downsamples

            y = self.pad_factor(y, y.size()[2:], factor)

        compression_output = self.prior.compress_forward(y, spatial_shape)

        attained_hbpp = 32 * len(compression_output.hyperlatents_encoded) / numpy.prod(spatial_shape)

        attained_lbpp = 32 * len(compression_output.latents_encoded) / numpy.prod(spatial_shape)

        attained_bpp = 32 * ((len(compression_output.hyperlatents_encoded) + len(
            compression_output.latents_encoded)) / numpy.prod(spatial_shape))

        if silent is False:
            self.logger.info("[ESTIMATED]")
            self.logger.info(f"BPP: {compression_output.total_bpp:.3f}")
            self.logger.info(f"HL BPP: {compression_output.hyperlatent_bpp:.3f}")
            self.logger.info(f"L BPP: {compression_output.latent_bpp:.3f}")

            self.logger.info("[ATTAINED]")
            self.logger.info(f"BPP: {attained_bpp:.3f}")
            self.logger.info(f"HL BPP: {attained_hbpp:.3f}")
            self.logger.info(f"L BPP: {attained_lbpp:.3f}")

        return compression_output

    def decompress(self, compression_output):
        assert self.model_mode == ModelMode.EVALUATION and (
                    self.training is False), f"Set model mode to {ModelMode.EVALUATION} for decompression."

        decoded = self.prior.decompress_forward(compression_output, device=get_device())

        reconstruction = self.generator(decoded)

        if self.normalize_input_image is True:
            reconstruction = torch.tanh(reconstruction)

        # Undo padding
        image_dims = compression_output.spatial_shape

        reconstruction = reconstruction[:, :, :image_dims[0], :image_dims[1]]

        if self.normalize_input_image is True:
            reconstruction = (reconstruction + 1.0) / 2.0

        return torch.clamp(reconstruction, 0.0, 1.0)

    def forward(
            self,
            x: torch.Tensor,
            generate: bool = False
    ) -> typing.Tuple[typing.Union[torch.Tensor, typing.Dict[str, torch.Tensor]], IntermediateData]:
        losses = {}

        if generate is True:
            self.step_counter += 1

        intermediate_data, hyperinfo = self.compression_forward(x)

        if self.model_mode == ModelMode.EVALUATION:
            reconstruction = intermediate_data.reconstruction

            if self.normalize_input_image is True:
                reconstruction = (reconstruction + 1.0) / 2.0

            return torch.clamp(reconstruction, 0.0, 1.0), intermediate_data

        compression_loss = self.compression_loss(intermediate_data, hyperinfo)

        if self.discriminate is True:
            discriminator_loss, generator_loss = self.generative_loss(intermediate_data, generate)

            weighted_generator_loss = self.beta * generator_loss

            compression_loss += weighted_generator_loss

            losses["discriminator_loss"] = discriminator_loss

        losses["compression_loss"] = compression_loss

        if self.step_counter % self.log_interval == 1:
            self.store_loss("weighted_compression_loss", compression_loss.item())

        return losses, intermediate_data
