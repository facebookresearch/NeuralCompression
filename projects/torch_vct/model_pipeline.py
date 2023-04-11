# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

## This module roughly corresponds to models.py in VCT.

import itertools
import math
from typing import Generator, List, NamedTuple, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import utils.memoize as memoize
from datamodules.video_data_api import Scenes, VideoData
from neural.entropy_model import PreviousLatent, VCTEntropyModel
from torch import Tensor

State = Tuple[PreviousLatent, ...]


class Bottleneck(NamedTuple):
    """
    Bottleneck of a single latent scene

    Args:
        latent_q: quantized latent, expected shape is [B, C, H, W]
        likelihood: likelihood/bits for the quantized latent, expected shape is
            [B', seq_len_dec, C] (no need to unpatch)
        entropy_model_features: optional, output of the entropy model, expected shape
            [B, d_model, H, W]
        -> by default, C=192, d_model=C*4=768
    """

    latent_q: Tensor
    likelihood: Tensor  # likelihood or bits
    entropy_model_features: Optional[Tensor] = None


class NetworkOut(NamedTuple):
    """Output of the entropy model decoder (.encode_and_decode_frames)

    Args:
        reconstruction: reconstruction of the latent, expected shape [B, C, H, W]
        likelihood: likelihood/bits of the reconstruction, expected shape
            [B', seq_len_dec, C] (no need to unpatch)
        -> by default, C=192
    """

    reconstruction: Tensor
    likelihood: Tensor


class PerChannelWeight(nn.Module):
    """Learn a weight per channel
    - used to get "fake" previous scene to encode the first one.
    """

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.weight = nn.parameter.Parameter(torch.rand((1, num_channels, 1, 1)))

    def forward(
        self, latent_shape: Union[torch.Size, List[int], Tuple[int, ...]]
    ) -> Tensor:
        assert latent_shape[1] == self.weight.shape[1], "channel length mismatch"
        return self.weight * self.weight.new_ones(latent_shape)


class ResidualBlock(nn.Module):
    """Standard residual block"""

    def __init__(self, filters: int, kernel_size: int) -> None:
        """Standard residual block

        Args:
            filters: conv filters, int
            kernel_size: kernel size, int
        """
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size, padding=1)
        self.activation = nn.LeakyReLU(negative_slope=0.01)  # default value
        self.conv2 = nn.Conv2d(filters, filters, kernel_size, padding=1)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Args:
            inputs: tensor of shape [B, C, H, W]

        Returns:
            tensor of shape [B, C, H, W]
        """
        output = self.activation(self.conv1(inputs))
        output = self.activation(self.conv2(output))
        return output + inputs


class Dequantizer(nn.Module):
    """Implement dequantization: feed y' = y + f(z) to the synthesis transform,
    where y is the latent and z is transformer/entropy model features,
    """

    def __init__(self, num_channels: int, d_model: int) -> None:
        """
        Feed y' = y + f(z) to the synthesis transform

        Args:
            num_channels: number of channels, int
            d_model: dimension of the model, int
        """
        super().__init__()
        self._d_model = d_model
        self._num_channels = num_channels
        self.process_conv = nn.Sequential(
            nn.Conv2d(d_model, num_channels, kernel_size=1),  # d_model, num_channels
            nn.LeakyReLU(negative_slope=0.01),
            ResidualBlock(num_channels, kernel_size=3),
        )

    def forward(
        self, *, latent_q: Tensor, entropy_features: Optional[Tensor] = None
    ) -> Tensor:
        """Calculates y'

        Args:
            latent_q: quantized latent, expected shape [B, C, H, W]
            entropy_features: optional tensor of features, expected shape
                [B, d_model, H, W]

        Returns:
            A tensor of shape [B, num_channels, H, W]
        """
        if entropy_features is None:
            # Create fake features with 0s only -- technically, this never gets hit
            b, _, h, w = latent_q.shape
            entropy_features = latent_q.new_zeros((b, self._d_model, h, w))

        return latent_q + self.process_conv(entropy_features)


class VCTPipeline(nn.Module):
    """
    Glue together encoder transform, entropy model and decoder transform
    """

    def __init__(
        self,
        analysis_transform: nn.Module,
        synthesis_transform: nn.Module,
        compression_channels: int,
        context_len: int = 2,
    ) -> None:
        """
        Setup overall model: encoder, entropy model and decoder
        """
        super().__init__()
        # Transforms
        self.analysis_transform = analysis_transform
        self.synthesis_transform = synthesis_transform
        self._pad_factor = 16

        assert (
            self.analysis_transform.compression_channels == compression_channels
        ), "Mismatched compression_channels"
        assert (
            self.synthesis_transform.compression_channels == compression_channels
        ), "Mismatched compression_channels"

        # Entropy model
        self.entropy_model = VCTEntropyModel(num_channels=compression_channels)
        self._temporal_pad_token_maker = PerChannelWeight(
            num_channels=compression_channels
        )
        self._dequantizer = Dequantizer(
            compression_channels, self.entropy_model.d_model
        )
        self._context_len = context_len

        self._code_to_strings = None

    ## ENCODING FUNCTIONS
    # I-latents
    def encode_Iscene(
        self, scene: Tensor, cache: memoize.Cache
    ) -> Tuple[State, Bottleneck]:
        """Encodes first scene/latent by creating a fake previous scene/latent

        Args:
            scene: single scene/latent tensor, expected shape [B, C, H, W], where C is
                equal to `compression_channels` that will be used by the `etropy_model`
            cache: Cache object

        Returns:
            tuple of two elements:
                - State object
                - Bottleneck object
        """
        # Create a fake previous latent and pass it through entropy model fwd pass
        fake_previous_latent = self._temporal_pad_token_maker(scene.shape)
        assert fake_previous_latent.shape == scene.shape
        processed = self.entropy_model.process_previous_latent_q(
            fake_previous_latent
        )  # PreviousLatent(quantized: Tensor, processed: Tensor)
        output = self.entropy_model(
            latent_unquantized=scene, previous_latents=(processed,)
        )  # TemporalEntropyModelOut object

        bottleneck = Bottleneck(
            output.perturbed_latent,  # [B, C, H, W]
            output.bits,  # [B', seq_len, C]
            output.features,  # [B, d_model, H, W]
        )
        decode_Iscene = memoize.bind(self.decode_Iscene, cache)
        _, state = decode_Iscene(bottleneck)
        return state, bottleneck

    # P-Frames
    def encode_Pscene(
        self,
        scene: Tensor,  # [B, C, H, W]
        scene_index: int,
        state: State,  # \hat y_t-1
        cache: memoize.Cache,
    ) -> Tuple[State, Bottleneck]:
        if not self.training and self._code_to_strings:
            # `compress` runs encoding and decoding
            output = self.entropy_model.compress(
                latent_unquantized=scene,
                previous_latents=state,
                # Since compression is lossless, decode only the first couple of latents
                # to check for errors, and skip decoding afterwards
                run_decode=scene_index < 3,
                validate_causal=False,
            )
        else:
            # forward pass through the entropy model, returns TemporalEntropyModelOut
            output = self.entropy_model(
                latent_unquantized=scene, previous_latents=state
            )  # [B, C, H, W], bits [b', seq_len^2, C], features [B, d_model, H, W]

        assert output.features is not None
        bottleneck = Bottleneck(
            latent_q=output.perturbed_latent,
            likelihood=output.bits,
            entropy_model_features=output.features,
        )

        # Run pscene decoder -- returns (reconstruction, new state)
        decode_Pscene = memoize.bind(self.decode_Pscene, cache)
        _, new_state = decode_Pscene(bottleneck=bottleneck, state=state, cache=cache)

        return new_state, bottleneck

    def encode_scenes(
        self, scenes: Scenes, cache: memoize.Cache
    ) -> Generator[Bottleneck, None, None]:
        """
        Args:
            scenes: Scenes object, containing `tensor` of shape [B, T, C, H, W]
            cache: Cache object
        """
        state = None
        for index, scene in enumerate(scenes.get_scenes_iter()):
            if index == 0:
                state, encode_out = self.encode_Iscene(scene, cache=cache)
            else:
                assert state is not None
                state, encode_out = self.encode_Pscene(scene, index, state, cache=cache)
            yield encode_out

    # DECODING
    # I-Frames (Scenes)
    @memoize.memoize
    def decode_Iscene(self, bottleneck: Bottleneck) -> Tuple[Tensor, State]:
        latent_q = bottleneck.latent_q  # [B, C, H, W]
        synthesis_in = self._dequantizer(
            latent_q=latent_q,  # .permute(0, 2, 3, 1).contiguous(),  # [B, H, W, C]
            entropy_features=bottleneck.entropy_model_features,
        )  # [B, C, H, W] this is the tensor we pass through image synthesis
        latent_q = bottleneck.latent_q  # noised
        latent_q = latent_q.detach()
        previous_latent = self.entropy_model.process_previous_latent_q(latent_q)
        # Note that this is a tuple, we start with a 1-length context.
        state: State = (previous_latent,)

        # NB: we apply the transforms (analysis and synthesis) in the forward pass onthe
        # entire video (TODO need to improve efficiency so as to handle long videos), so
        # here we return synthesis_in instead of reconstruction and apply image_synthesis
        return synthesis_in, state

    # P-Frames (Scenes)
    @memoize.memoize
    def decode_Pscene(
        self, bottleneck: Bottleneck, state: State, cache: memoize.Cache
    ) -> Tuple[Tensor, State]:
        latent_q = bottleneck.latent_q  # [B, C, H, W]
        synthesis_in = self._dequantizer(
            latent_q=latent_q, entropy_features=bottleneck.entropy_model_features
        )  # [B, H, W, C] this is the tensor we pass through image synthesis

        # Preprocess current quantized latent, `latent_q`.
        next_state_entry = self.entropy_model.process_previous_latent_q(latent_q)
        new_state = (*state, next_state_entry)
        new_state = new_state[0 - self._context_len :]
        # NB: as in decode_Iscene, we return synthesis_in instead of reconstruction and
        # apply image_synthesis in the forward pass
        return synthesis_in, new_state

    # All frames (scenes)
    def decode_scenes(
        self, bottlenecks: Generator[Bottleneck, None, None], cache: memoize.Cache
    ) -> Generator[Tensor, None, None]:
        decode_Iscene = memoize.bind(self.decode_Iscene, cache)
        decode_Pscene = memoize.bind(self.decode_Pscene, cache)

        state = None
        for index, bottleneck in enumerate(bottlenecks):
            if index == 0:
                scene_reconstruction, state = decode_Iscene(bottleneck)
            else:
                assert state is not None
                scene_reconstruction, state = decode_Pscene(
                    bottleneck, state=state, cache=cache
                )
            yield scene_reconstruction

    def encode_and_decode_scenes(
        self, scenes: Scenes, cache: memoize.Cache
    ) -> Generator[NetworkOut, None, None]:
        """Encodes and decodes latents/scenes

        Args:
            scenes: Scenes object, containing `tensor` attribute of shape [B, T, C, H, W]
            cache: Cache object

        Yields:
            `NetworkOut` object containng reconstruction of the scene/latent and
                likelihoods
        """
        encode_outs = self.encode_scenes(scenes, cache=cache)

        # Iterate over `encode_outs` twice: once to decode and once to construct and
        # yield the NetworkOut object
        encode_outs, encode_outs_tee = itertools.tee(encode_outs)
        reconstructions_scenes = self.decode_scenes(
            (encode_out for encode_out in encode_outs_tee), cache=cache
        )  # [B, C, H, W]

        for rec_scene, encode_out in zip(reconstructions_scenes, encode_outs):
            # rec_scene is padded [B, C, H, W]
            yield NetworkOut(
                reconstruction=rec_scene,
                likelihood=encode_out.likelihood,
            )

    ### FORWARD & HELPERS ###
    def _pad(
        self, x: Tensor, sizes: Sequence[int], factor: Optional[int] = None
    ) -> Tensor:
        """
        Args:
            x: tensor of size [B, T, C, H, W] to pad
            sizes: height and width of x
            factor: optional, if provided this is the factor to pad x to.
                Defaults to None.
        Returns:
            padded tensor of size [B, T, C, H+pad_h, W+pad_w]
        """
        if factor is None:
            n_im_downscale = getattr(
                self.analysis_transform, "num_downsampling_layers", 0
            )
            n_hyper_downscale = getattr(
                self.entropy_model, "num_downsampling_layers", 0
            )
            factor = 2 ** (n_im_downscale + n_hyper_downscale)

        pad_h, pad_w = [(factor - (s % factor)) % factor for s in sizes]  # type: ignore
        # dims are in reverse -- W, H, C, so the below pads:
        # width dimension by 0, pad_w, height by 0, pad_h and no padding for channel
        return torch.nn.functional.pad(x, (0, pad_w, 0, pad_h, 0, 0), "reflect")

    def forward(self, video: VideoData) -> Tuple[Tensor, list]:
        """
        Args:
            video: a batch of video clips with shape [B, T, C, H, W], where B is batch
                size, T is number of frames in the video, C=3 (RGB).

        Returns:
            A tuple of:
                2. reconstructions_frames: reconstruction of the 'decompressed' video.
                    Shape is same as the original, [B, T, C, H, W].
                5. bottleneck_output: a list (possibly empty) containing any additional
                    outputs from the bottleneck module, e.g. to compute rate.
        """
        video.validate_shape()
        if not self.training:
            video_shape = video.spatial_shape
            inputs = self._pad(video.video_tensor, video_shape)
        else:
            video_shape = None
            inputs = video.video_tensor

        scenes = Scenes(self.analysis_transform(inputs))  # [B, T, C_emb, H_emb, W_emb]

        cache = memoize.create_cache()

        # code and decode in a differentiable way
        res = self.encode_and_decode_scenes(scenes, cache=cache)  # yields NetworkOut
        rec_scenes = []
        likelihoods = []
        for r in res:
            rec_scenes.append(r.reconstruction)
            likelihoods.append(r.likelihood)

        reconstructions_embeddings = torch.stack(rec_scenes, dim=1)  # [B, T, C, H, W]
        likelihoods = torch.stack(likelihoods)  # [T, ...]

        reconstructions_frames = self.synthesis_transform(
            reconstructions_embeddings, video.shape
        )
        if not self.training:
            assert video_shape is not None, "frames_shape not found"
            h, w = video_shape
            reconstructions_frames = reconstructions_frames[..., :h, :w]

        return (
            reconstructions_frames,
            [likelihoods],  # bottleneck output
        )

    def _on_cpu(self) -> bool:
        cpu = torch.device("cpu")
        for param in self.parameters():
            if param.device != cpu:
                return False
        return True

    def compute_rate(
        self, scenes_likelihoods: Tensor, per_frame: bool = False
    ) -> Tensor:
        if per_frame:
            bits = (scenes_likelihoods.log().sum(dim=(1, 2, 3))) / -math.log(2.0)
        else:
            bits = (scenes_likelihoods.log().sum()) / -math.log(2.0)

        return bits

    def _prepare_for_compression(self):
        self.eval()
        self.entropy_model.update(force=False)
        self._code_to_strings = True
        assert self.training is False

    def compress_video(
        self, video: VideoData, force_cpu: bool = False
    ) -> Tuple[Tensor, List]:
        self._prepare_for_compression()

        assert (
            video.video_tensor.dim() == 5 and video.video_tensor.shape[0] == 1
        ), f"expected batch 1 video of shape [1, T, 3, H, W], got {video.shape}"

        if not self._on_cpu() and force_cpu:
            raise ValueError("Compression not supported on GPU.")

        frames_shape = video.spatial_shape
        x = self._pad(video.video_tensor, frames_shape)
        x = Scenes(self.analysis_transform(x))

        # `encode_and_decode_scenes` compresses to strings if training we are in eval
        # mode and _code_to_strings is set to True
        network_outs = self.encode_and_decode_scenes(x, cache=None)  # no cache in eval
        rec_scenes = []
        bits = []
        for i, res in enumerate(network_outs):
            rec_scenes.append(res.reconstruction)  # [1, C, H, W]
            if i == 0:
                assert res.likelihood.dim() == 3
                bits.append(self.compute_rate(res.likelihood).item())
            else:
                assert res.likelihood.dim() == 1
                bits.append(res.likelihood.item())

        # Stack recosntructions to get [B, T, C, H, W]; sum to get total bits
        return torch.stack(rec_scenes, dim=1), bits

    def decompress_video(
        self,
        frames_shape,
        bottleneck_args: Sequence,  # [rec_scenes] of shape [B, T, C, H, W]
        force_cpu: bool = False,
    ) -> Tensor:
        # TODO We might want to do the deccoding here (it is doen in compress_video)
        # Here we only apply image synthesis
        if not self._on_cpu() and force_cpu:
            raise ValueError("Decompression not supported on GPU.")

        reconstructions_frames = self.synthesis_transform(
            bottleneck_args[0], frames_shape
        )

        assert len(frames_shape) == 5
        return reconstructions_frames
