# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Code is based on the papers:

Lu, Guo, et al. "DVC: An end-to-end deep video compression framework."
CVPR (2019).

Yang, Ren et al.
"OpenDVC: An Open Source Implementation of the DVC Video Compression Method"
arXiv:2006.15862
"""
from typing import Any, List, NamedTuple, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from compressai.entropy_models import EntropyBottleneck
from torch import Tensor, nn

import neuralcompression.functional as ncF
from neuralcompression.layers import SimplifiedGDN, SimplifiedInverseGDN


class CompressedPFrame(NamedTuple):
    """
    Output of DVC model compressing a single frame.

    Args:
        compressed_flow: Flow fields compressed to a string.
        flow_decomp_sizes: Metadata for the size of the flow representations at
            each compression level.
        compressed_residual: Residual compressed to a string.
        residual_decomp_sizes: Metadata for the size of the residual
            representations at each compression level.
    """

    compressed_flow: Any
    flow_decomp_sizes: List[torch.Size]
    compressed_residual: Any
    residual_decomp_sizes: List[torch.Size]


class DVCTrainOutput(NamedTuple):
    """
    Output of DVC forward function.

    Args:
        flow: The calculated optical flow field.
        image2_est: An estimate of image2 using the flow field and residual.
        residual: The residual between the flow-compensated ``image1`` and the
            true ``image2``.
        flow_probabilities: Estimates of probabilities of each compressed flow
            latent using entropy bottleneck.
        resid_probabilities: Estimates of probabilities of each compressed
            residual latent using entropy bottleneck.
    """

    flow: Tensor
    image2_est: Tensor
    residual: Tensor
    flow_probabilities: Tensor
    resid_probabilities: Tensor


class DVC(nn.Module):
    """
    Deep Video Compression composition class.

    This composes the Deep Video Compression Module of Lu (2019). It requires
    defining a motion estimator, a motion autoencoder, a motion compensator,
    and a residual autoencoder. The individual modules can be input by the
    user. If the user does not input a module, then this class will construct
    the module with default parameters from the paper.

    Lu, Guo, et al. "DVC: An end-to-end deep video compression framework."
    CVPR (2019).

    Args:
        coder_channels: Number of channels to use for autoencoders.
        motion_estimator: A module for estimating motion. See
            ``DVCPyramidFlowEstimator`` for an example.
        motion_encoder: A module for encoding motion fields. See
            ``DVCCompressionEncoder`` for an example.
        motion_entropy_bottleneck: A module for quantization and latent
            probability estimation for the compressed motion fields. See
            ``EntropyBottleneck`` for an example.
        motion_decoder: A module for decoding motion fields. See
            ``DVCCompressionDecoder`` for an example.
        motion_compensation: A module for compensating for motion errors. See
            ``DVCMotionCompensationModel`` for an example.
        residual_encoder: A module for encoding residuals after motion
            compensation. See ``DVCCompressionEncoder`` with 3 input channels
            for an example.
        residual_entropy_bottleneck: A module for quantization and latent
            probability estimation for the compressed residuals. See
            ``EntropyBottleneck`` for an example.
        residual_decoder: A module for decoding residuals. See
            ``DVCCompressionEncoder`` with 3 output channels for an example.
    """

    def __init__(
        self,
        coder_channels: int = 128,
        motion_estimator: Optional[nn.Module] = None,
        motion_encoder: Optional[nn.Module] = None,
        motion_entropy_bottleneck: Optional[nn.Module] = None,
        motion_decoder: Optional[nn.Module] = None,
        motion_compensation: Optional[nn.Module] = None,
        residual_encoder: Optional[nn.Module] = None,
        residual_entropy_bottleneck: Optional[nn.Module] = None,
        residual_decoder: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.motion_estimator = (
            DVCPyramidFlowEstimator() if motion_estimator is None else motion_estimator
        )
        self.motion_encoder = (
            DVCCompressionEncoder(
                filter_channels=coder_channels, out_channels=coder_channels
            )
            if motion_encoder is None
            else motion_encoder
        )
        self.motion_entropy_bottleneck = (
            EntropyBottleneck(coder_channels)
            if motion_entropy_bottleneck is None
            else motion_entropy_bottleneck
        )
        self.motion_decoder = (
            DVCCompressionDecoder(
                in_channels=coder_channels, filter_channels=coder_channels
            )
            if motion_decoder is None
            else motion_decoder
        )
        self.motion_compensation = (
            DVCMotionCompensationModel()
            if motion_compensation is None
            else motion_compensation
        )
        self.residual_encoder = (
            DVCCompressionEncoder(
                in_channels=3,
                filter_channels=coder_channels,
                out_channels=coder_channels,
                kernel_size=5,
            )
            if residual_encoder is None
            else residual_encoder
        )
        self.residual_entropy_bottleneck = (
            EntropyBottleneck(coder_channels)
            if residual_entropy_bottleneck is None
            else residual_entropy_bottleneck
        )
        self.residual_decoder = (
            DVCCompressionDecoder(
                in_channels=coder_channels,
                filter_channels=coder_channels,
                out_channels=3,
                kernel_size=5,
            )
            if residual_decoder is None
            else residual_decoder
        )

    def compress(self, image1: Tensor, image2: Tensor) -> CompressedPFrame:
        """
        Compute compressed motion and residual between two images.

        Note:
            You must call ``update`` prior to calling ``compress``.

        Args:
            image1: The first image.
            image2: The second image.

        Returns:
            CompressedPFrame tuple with 1) the compressed flow field, 2) the
            flow decomposition sizes, 3) the compressed residual, and 4) the
            residual decomposition sizes.
        """
        flow = self.motion_estimator(image1, image2)

        # compress optical flow fields
        flow_latent, flow_decomp_sizes = self.motion_encoder(flow)
        compressed_flow = self.motion_entropy_bottleneck.compress(flow_latent)
        flow_latent = self.motion_entropy_bottleneck.decompress(
            compressed_flow, flow_latent.shape[-2:]
        )
        flow = self.motion_decoder(flow_latent, flow_decomp_sizes)

        # apply optical flow fields
        image2_est = ncF.dense_image_warp(image1, flow.permute(0, 2, 3, 1))

        # compensate for optical flow errors
        image2_est = self.motion_compensation(image1, image2_est, flow)

        # encode final residual
        residual = image2 - image2_est
        residual_latent, resid_decomp_sizes = self.residual_encoder(residual)
        compressed_residual = self.residual_entropy_bottleneck.compress(residual_latent)

        return CompressedPFrame(
            compressed_flow,
            flow_decomp_sizes,
            compressed_residual,
            resid_decomp_sizes,
        )

    def decompress(self, image1: Tensor, compressed_pframe: CompressedPFrame) -> Tensor:
        """
        Decompress motion fields and residual and compute next frame estimate.

        Args:
            image1: The base image for computing the next frame estimate.
            compressed_pframe: A compressed P-frame and metadata.

        Returns:
            An estimate of ``image2`` using ``image1`` and the compressed
                transition information.
        """
        flow_latent_size = (
            compressed_pframe.flow_decomp_sizes[-1][-2] // 2,
            compressed_pframe.flow_decomp_sizes[-1][-1] // 2,
        )
        flow_latent = self.motion_entropy_bottleneck.decompress(
            compressed_pframe.compressed_flow, flow_latent_size
        )
        flow = self.motion_decoder(flow_latent, compressed_pframe.flow_decomp_sizes)

        # apply optical flow fields
        image2_est = ncF.dense_image_warp(image1, flow.permute(0, 2, 3, 1))

        # compensate for optical flow errors
        image2_est = self.motion_compensation(image1, image2_est, flow)

        # decode residual
        residual_latent_size = (
            compressed_pframe.residual_decomp_sizes[-1][-2] // 2,
            compressed_pframe.residual_decomp_sizes[-1][-1] // 2,
        )
        residual_latent = self.motion_entropy_bottleneck.decompress(
            compressed_pframe.compressed_residual, residual_latent_size
        )
        residual = self.residual_decoder(
            residual_latent, compressed_pframe.residual_decomp_sizes
        )

        return (image2_est + residual).clamp_(0, 1)

    def update(self, force: bool = True) -> bool:
        """
        Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            force: Overwrite previous values.
        """
        update1 = self.motion_entropy_bottleneck.update(force=force)
        update2 = self.residual_entropy_bottleneck.update(force=force)

        return update1 or update2

    def forward(self, image1: Tensor, image2: Tensor) -> DVCTrainOutput:
        """
        Apply DVC coding to a pair of images.

        The ``forward`` function is expected to be used for training. For
        inference, see the ``compress`` and ``decompress``.

        Args:
            image1: The base image.
            image2: The second image. Optical flow will be applied to
                ``image1`` to predict ``image2``.

        Returns:
            A 5-tuple training output containing 1) the calculated optical
            flow, 2) an estimate of ``image2`` including motion compensation,
            3) the residual between ``image2`` and its estimate, 4) the
            probabilities output by the flow quantizer, and 5) the
            probabilities output by the residual quantizer.
        """
        # estimate optical flow fields
        flow = self.motion_estimator(image1, image2)

        # compress optical flow fields
        flow, sizes = self.motion_encoder(flow)
        flow, flow_probabilities = self.motion_entropy_bottleneck(flow)
        flow = self.motion_decoder(flow, sizes)

        # apply optical flow fields
        image2_est = ncF.dense_image_warp(image1, flow.permute(0, 2, 3, 1))

        # compensate for optical flow errors
        image2_est = self.motion_compensation(image1, image2_est, flow)

        # encode final residual
        residual = image2 - image2_est
        residual, sizes = self.residual_encoder(residual)
        residual, resid_probabilities = self.residual_entropy_bottleneck(residual)
        residual = self.residual_decoder(residual, sizes)

        image2_est = image2_est + residual

        return DVCTrainOutput(
            flow, image2_est, residual, flow_probabilities, resid_probabilities
        )


class DVCCompressionDecoder(nn.Module):
    """
    Deep Video Compression decoder module.

    Args:
        in_channels: Number of channels in latent space.
        out_channels: Number of channels to output from decoder.
        filter_channels: Number of channels for intermediate layers.
        kernel_size: Size of convolution kernels.
        stride: Stride of convolutions.
        num_conv_layers: Number of convolution layers.
        use_gdn: Whether to use Generalized Divisive Normalization or
            BatchNorm.
    """

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 2,
        filter_channels: int = 128,
        kernel_size: int = 3,
        stride: int = 2,
        num_conv_layers: int = 4,
        use_gdn: bool = True,
    ):
        super().__init__()
        padding = kernel_size // 2
        bias = True if use_gdn else False

        self.layers = nn.ModuleList()
        for _ in range(num_conv_layers - 1):
            self.layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=filter_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=bias,
                )
            )
            if use_gdn:
                self.layers.append(SimplifiedInverseGDN(filter_channels))
            else:
                self.layers.append(nn.BatchNorm2d(filter_channels))
                self.layers.append(nn.ReLU())

            in_channels = filter_channels

        self.layers.append(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=True,
            )
        )

    def forward(
        self, image: Tensor, output_sizes: Optional[Sequence[torch.Size]] = None
    ) -> Tensor:
        # validate output_sizes
        if output_sizes is not None:
            output_sizes = [os for os in output_sizes]  # shallow copy
            transpose_conv_count = 0
            for layer in self.layers:
                if isinstance(layer, nn.ConvTranspose2d):
                    transpose_conv_count += 1
            if not transpose_conv_count == len(output_sizes):
                raise ValueError(
                    "len(output_sizes) must match number of transpose convolutions."
                )

        # run the deconvolutions
        for layer in self.layers:
            # use sizes from encoder if we have them for decoding
            if isinstance(layer, nn.ConvTranspose2d) and output_sizes is not None:
                image = layer(image, output_sizes.pop())
            else:
                image = layer(image)

        return image


class DVCCompressionEncoder(nn.Module):
    """
    Deep Video Compression encoder module.

    Args:
        in_channels: Number of channels in image space.
        out_channels: Number of channels to output from encoder.
        filter_channels: Number of channels for intermediate layers.
        kernel_size: Size of convolution kernels.
        stride: Stride of convolutions.
        num_conv_layers: Number of convolution layers.
        use_gdn: Whether to use Generalized Divisive Normalization or
            BatchNorm.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 128,
        filter_channels: int = 128,
        kernel_size: int = 3,
        stride: int = 2,
        num_conv_layers: int = 4,
        use_gdn: bool = True,
    ):
        super().__init__()
        padding = kernel_size // 2
        bias = True if use_gdn else False

        self.layers = nn.ModuleList()
        for _ in range(num_conv_layers - 1):
            self.layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=filter_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=bias,
                )
            )
            if use_gdn:
                self.layers.append(SimplifiedGDN(filter_channels))
            else:
                self.layers.append(nn.BatchNorm2d(filter_channels))
                self.layers.append(nn.ReLU())

            in_channels = filter_channels

        self.layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=True,
            )
        )

    def forward(self, image: Tensor) -> Tuple[Tensor, List[torch.Size]]:
        sizes = []
        for layer in self.layers:
            # log the sizes so that we can recover them with strides != 1
            if isinstance(layer, nn.Conv2d):
                sizes.append(image.shape)

            image = layer(image)

        return image, sizes


class DVCPyramidFlowEstimator(nn.Module):
    """
    Pyramidal optical flow estimation.

    This estimates the optical flow at `levels` different pyramidal scales. It
    should be trained in conjunction with a pyramidal optical flow supervision
    signal.

    Args:
        in_channels: Number of input channels. Typically uses two 3-channel
            images, plus an initial flow with 2 more channels.
        filter_counts: Number of filters for each stage of pyramid. Defaults
            to ``[32, 64, 32, 16, 2]``.
        kernel_size: Kernel size of filters.
        levels: Number of pyramid levels.
    """

    def __init__(
        self,
        in_channels: int = 8,
        filter_counts: Optional[Sequence[int]] = None,
        kernel_size: int = 7,
        levels: int = 5,
    ):
        super().__init__()
        if filter_counts is None:
            filter_counts = [32, 64, 32, 16, 2]
        padding = kernel_size // 2
        self.model_levels = nn.ModuleList()

        for _ in range(levels):
            current_in_channels = in_channels
            layers: List[nn.Module] = []
            for i, filter_count in enumerate(filter_counts):
                layers.append(
                    nn.Conv2d(
                        in_channels=current_in_channels,
                        out_channels=filter_count,
                        kernel_size=kernel_size,
                        padding=padding,
                    )
                )
                if i < len(filter_counts) - 1:
                    layers.append(nn.ReLU())

                current_in_channels = filter_count

            self.model_levels.append(nn.Sequential(*layers))

    def _decompose_images_to_pyramids(
        self, image1: Tensor, image2: Tensor
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Pyramid flow structure, average down for each pyramid level."""
        images_1 = [image1]
        images_2 = [image2]
        for _ in range(1, len(self.model_levels)):
            images_1.append(F.avg_pool2d(images_1[-1], 2))
            images_2.append(F.avg_pool2d(images_2[-1], 2))

        return images_1, images_2

    def calculate_flow_with_image_pairs(
        self, image1: Tensor, image2: Tensor
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        """
        Calculate optical flow and return image pairs for each pyramid level.

        During training, we optimize the optical flow by trying to match the
        flow at each pyramid scale to its target. This function computes the
        images output by the flow at each level of the pyramid and returns them
        all in addition to the final flow.

        Args:
            image1: The based image - a flow map will be computed to transform
                ``image1`` into ``image2``.
            image2: The target image.

        Returns:
            A 2-tuple containing:
                The pyramidal flow estimate.
                Base/target image pairs from each level of the pyramid that can
                    be input into a distortion function.
        """
        if not image1.shape == image2.shape:
            raise ValueError("Image shapes must match.")

        images_1, images_2 = self._decompose_images_to_pyramids(image1, image2)
        flow = torch.zeros_like(images_1[-1])[:, :2]  # just the first two channels

        # estimate flows at all levels and return warped images
        optical_flow_pairs = []
        for model_level in self.model_levels:
            image1 = images_1.pop()
            image2 = images_2.pop()
            flow = F.interpolate(flow, image1.shape[2:], mode="bilinear")
            flow = flow + model_level(
                torch.cat(
                    (
                        ncF.dense_image_warp(image1, flow.permute(0, 2, 3, 1)),
                        image2,
                        flow,
                    ),
                    dim=1,
                )
            )
            optical_flow_pairs.append(
                (ncF.dense_image_warp(image1, flow.permute(0, 2, 3, 1)), image2)
            )

        # we have to return all of the warped images at different levels for
        # training
        return flow, optical_flow_pairs

    def forward(self, image1: Tensor, image2: Tensor) -> Tensor:
        """
        Calculate optical flow using pyramidal structure and a learned model.

        Args:
            image1: The based image - a flow map will be computed to transform
                ``image1`` into ``image2``.
            image2: The target image.

        Returns:
            Pyramidal optical flow estimate.
        """
        return self.calculate_flow_with_image_pairs(image1, image2)[0]


class DVCMotionCompensationModel(nn.Module):
    """
    Motion compensation model.

    After applying optical flow, there remain residual errors. This model
    corrects for the errors based on the input image, the transformed image,
    and the optical flow field.

    Args:
        in_channels: Number of input channels. The ``forward`` function takes
            as input two 3-channel images, plus an optical flow field, so by
            default this is 8 channels.
        model_channels: Number of channels for convolution kernels.
        out_channels: Number of channels for output motion-compensated image.
        kernel_size: Size of convolution kernels.
        num_levels: Number of convolution levels.
    """

    def __init__(
        self,
        in_channels: int = 8,
        model_channels: int = 64,
        out_channels: int = 3,
        kernel_size: int = 3,
        num_levels: int = 3,
    ):
        super().__init__()
        padding = kernel_size // 2

        # simple input layer
        input_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=model_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

        # build the autoencoder portion from the ground up
        child = None
        for _ in range(num_levels):
            child = _MotionCompensationLayer(
                child, in_channels=model_channels, out_channels=model_channels
            )
        unet = child

        # make sure we turn on bias for the output
        output_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=model_channels,
                out_channels=model_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=True,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=model_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=True,
            ),
        )

        assert unet is not None
        self.model = nn.Sequential(input_layer, unet, output_layer)

    def forward(self, image1: Tensor, image2: Tensor, flow: Tensor) -> Tensor:
        return self.model(torch.cat((image1, image2, flow), dim=1))


class _MotionCompensationLayer(nn.Module):
    """
    Intermediate U-Net style layer for motion compensation.

    Args:
        child: The motion compensation module executes a U-Net style
            convolution structure with an input convolution followed by a child
            ``_MotionCompensationLayer`` followed by an output convolution. If
            the layer is at the bottom of the 'U', then a ``None`` is input for
            ``child``.
        in_channels: Number of input channels for convolutions.
        out_channels: Number of outpout channels for convolutions.
    """

    def __init__(
        self,
        child: Optional[nn.Module] = None,
        in_channels: int = 64,
        out_channels: int = 64,
    ):
        super().__init__()
        self.child = child
        self.input_block = _ResidualBlock(
            in_channels=in_channels, out_channels=out_channels
        )
        self.output_block = _ResidualBlock(
            in_channels=in_channels, out_channels=out_channels
        )

    def forward(self, image: Tensor) -> Tensor:
        upsample_size = tuple(image.shape[2:])

        # residual block includes skipped connection
        image = self.input_block(image)

        # if we have a child, downsample, run it, and then upsample back
        if self.child is not None:
            image = image + F.interpolate(
                self.child(F.avg_pool2d(image, kernel_size=2)),
                size=upsample_size,
                mode="nearest",
            )

        # residual block includes skipped connection
        return self.output_block(image)


class _ResidualBlock(nn.Module):
    """
    Simple pre-activated ResNet-style block.

    Structure: input -> ReLU -> Conv2d -> ReLU -> Conv2d + input

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride of convolutions.
    """

    def __init__(
        self,
        kernel_size: int = 3,
        in_channels: int = 64,
        out_channels: int = 64,
        stride: int = 1,
    ):
        super().__init__()
        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
            ),
        )

    def forward(self, image: Tensor) -> Tensor:
        return image + self.block(image)
