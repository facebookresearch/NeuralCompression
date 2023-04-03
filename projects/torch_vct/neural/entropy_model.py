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

import itertools
import sys
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .bottlenecks import GsnConditionalLocScaleShift
from .entropy_model_layers import EncoderSection, LearnedPosition, StartSym, Transformer
from .layers_utils import make_embedding
from .patcher import Patcher

_LATENT_NORM_FAC: float = 35.0  # factor to scale latents by


class PreviousLatent(NamedTuple):
    """Previous latent with the following attributes

    Attributes:
        quantized: the quantized latent
        processed: the processed latent by running it through an encoder. See
            `VCTEntropyModle.process_previous_latent_q` for more details.
    """

    quantized: Tensor
    processed: Tensor


class TemporalEntropyModelOut(NamedTuple):
    """Output of the VCT temporal entropy model

    Attributes:
      perturbed_latent: noised (training=True) or quantized (training=False) latent.
        Tensor of shape [b', seq_len, C]
      bits: bits taken to transmit the latent. Tensor of shape: [b', seq_len, C]
      features: (optional) features of the entropy model to be used by a synthesis
        transform for dequantizing. Tensor of shape: [B, d_model, H, W]
    """

    perturbed_latent: Tensor
    bits: Tensor
    features: Optional[Tensor] = None


class VCTEntropyModel(nn.Module):
    """
    Temporal Entropy Model
    """

    def __init__(
        self,
        num_channels: int = 192,
        context_len: int = 2,
        window_size_enc: int = 8,
        window_size_dec: int = 4,
        num_layers_encoder_sep: int = 3,
        num_layers_encoder_joint: int = 2,
        num_layers_decoder: int = 5,
        d_model: int = 768,
        num_head: int = 16,
        mlp_expansion: int = 4,
        drop_out_enc: float = 0.0,
        drop_out_dec: float = 0.0,
    ) -> None:
        """
        Temporal Entropy Model
        Args:
            num_channels: number of channels in the latent space,
                i.e. symbols per token. Defaults to 196.
            context_len: number of previous latents. Defaults to 2.
            window_size_enc: window (patch) size in encoder.
                Defaults to 8.
            window_size_dec: window (patch) size in decoder.
                Defaults to 4.
            num_layers_encoder_sep: number of layers in the separate encoder.
                Defaults to 3.
            num_layers_encoder_joint: number of layers in the joint encoder.
                Defaults to 2.
            num_layers_decoder: number of layers in the decoder.
                Defaults to 5.
            d_model: feature dimensionality inside the model.
                Defaults to 768.
            num_head: number of attention heads in MHA layers.
                Defaults to 16.
            mlp_expansion: expansion *factor* for each MLP.
                Defaults to 4.
            drop_out_enc: dropout probability in encoder.
                Defaults to 0.0.
            drop_out_dec: dropout probability in decoder.
                Defaults to 0.0.
        """
        super().__init__()
        if window_size_enc < window_size_dec:
            raise ValueError(
                f"window_size_enc={window_size_enc} cannot be lower"
                f"than window_size_dec={window_size_dec}."
            )
        if num_channels < 0:
            raise ValueError(f"num_channels={num_channels} cannot be negative")
        self.num_channels = num_channels
        self.window_size_enc = window_size_enc
        self.window_size_dec = window_size_dec

        self.d_model = d_model
        # we will use compressai's GsnConditional as a bottleneck
        self.bottleneck = GsnConditionalLocScaleShift(
            num_scales=256, num_means=100, min_scale=0.01, tail_mass=(2 ** (-8))
        )

        self.range_bottleneck = None

        self.context_len = context_len
        self.encoder_sep = EncoderSection(
            num_layers=num_layers_encoder_sep,
            num_heads=num_head,
            hidden_dim=d_model,
            mlp_expansion=mlp_expansion,
            dropout=drop_out_enc,
        )
        self.encoder_joint = EncoderSection(
            num_layers=num_layers_encoder_joint,
            num_heads=num_head,
            hidden_dim=d_model,
            mlp_expansion=mlp_expansion,
            dropout=drop_out_enc,
        )

        self.seq_len_dec = window_size_dec**2
        self.seq_len_enc = window_size_enc**2

        self.patcher = Patcher(window_size_dec, "reflect")
        self.learned_zero = StartSym(hidden_dim=num_channels)

        self.enc_position_sep = LearnedPosition(
            seq_length=self.seq_len_enc, hidden_dim=d_model
        )
        self.enc_position_joint = LearnedPosition(
            seq_length=self.seq_len_enc * context_len, hidden_dim=d_model
        )

        self.dec_position = LearnedPosition(
            seq_length=self.seq_len_dec, hidden_dim=d_model
        )

        self.post_embedding_layernorm = nn.LayerNorm(d_model, eps=1e-6)

        self.encoder_embedding = make_embedding(
            input_dim=num_channels, hidden_dim=d_model
        )  # a single linear layer
        self.decoder_embedding = make_embedding(
            input_dim=num_channels, hidden_dim=d_model
        )  # a single linear layer

        self.decoder = Transformer(
            seq_length=self.seq_len_dec,
            num_layers=num_layers_decoder,
            num_heads=num_head,
            hidden_dim=d_model,
            mlp_expansion=mlp_expansion,
            dropout=drop_out_dec,
            is_decoder=True,
        )  # its forward method returns [B', seq_length, hidden_dim]

        def _make_final_heads(output_channels: int) -> nn.Module:
            # 3 stacked linear layers with leakyrelu activations
            return nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, output_channels),
            )

        self.mean_head = _make_final_heads(num_channels)
        self.scale_head = _make_final_heads(num_channels)

    @staticmethod
    def round_st(x: Tensor) -> Tensor:
        """
        Straight-trhough round
        """
        return (torch.round(x) - x).detach() + x

    def process_previous_latent_q(
        self, previous_latent_quantized: Tensor
    ) -> PreviousLatent:
        """Process previous quantized latent by passing it through the encoder.

        This can be used if previous latents go through expensive transforms
        before being fed to the entropy model, and will be stored in the `processed`
        field of the `PreviousLatent` tuple.

        The output of this function applied to all quantized latents should
        be fed to the `forward` method. This is used to improve efficiency,
        as it avoids calling expensive processing of previous latents at
        each time step.

        Args:
            previous_latent_quantized: previous quantized latent that is to be processed,
                expected shape [B, C, H, W]

        Returns:
            PreviousLatent object with the processed latent in the processed field
        """
        patches, _ = self.patcher(
            previous_latent_quantized, self.window_size_enc
        )  # [b', seq_len, C], seq_len = patch_size^2, C=num_channels

        patches = patches / _LATENT_NORM_FAC
        patches = self.encoder_embedding(patches)  # [b', seq_len, d_model]
        patches = self.post_embedding_layernorm(patches)  # [b', seq_len, d_model]
        patches = self.enc_position_sep(patches)  # [b', seq_len, d_model]
        patches = self.encoder_sep(patches)  # [b', seq_len, d_model]

        return PreviousLatent(previous_latent_quantized, processed=patches)

    def _embed_latent_q_patched(self, latent_q_patched: Tensor) -> Tensor:
        """Embed current patched latent for decoder

        The input latent is normalized, embedded in d_model dimension, and
        positional encoding is added.
        Args:
            latent_q_patched: tensor of shape [b', seq_len_dec, C]

        Returns:
            tensor of shape [b', seq_len_dec, d_model]
        """
        latent_q_patched = latent_q_patched / _LATENT_NORM_FAC  # [b', seq_len_dec, C]
        latent_q_patched = self.decoder_embedding(
            latent_q_patched
        )  # [b', seq_len_dec, d_model]
        latent_q_patched = self.post_embedding_layernorm(
            latent_q_patched
        )  # [b', seq_len_dec, d_model]
        return self.dec_position(latent_q_patched)  # [b', seq_len_dec, d_model]

    def _get_transformer_output(
        self, *, encoded_patched: Tensor, latent_q_patched: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Predict the distribution of the current quantized patched latent

        Args:
            encoded_patched: tensor of shape [*b, context_len*patch_enc^2, d_model],
                where with the defaults (patch_enc=8, d_model=768), so we have
                default expected shape [*b, 128, 768]
            latent_q_patched: tensor of shape [*b, patch_dec^2, num_channels], where
                with the default patch_dec^2=16 and num_channels = 192, so we have
                default expected shape [*b, 16, 192]

        Returns:
            a tuple containing 3 tensors: mean, scale and decoder output
        """
        if encoded_patched.shape[-1] != self.d_model:
            raise ValueError(
                f"Context must have final dim {self.d_model}, "
                f"got shape={encoded_patched.shape}. "
                "Did you run `process_previous_latent_q`?"
            )
        latent_q_patched_shifted = self.learned_zero(latent_q_patched)
        del latent_q_patched  # should not be used after this line

        latent_q_patched_emb_shifted = self._embed_latent_q_patched(
            latent_q_patched_shifted
        )  # [B', seq_length_enc, d_model]

        encoded_patched = self.enc_position_joint(encoded_patched)
        encoded_patched = self.encoder_joint(encoded_patched)  # [B', seq_len, d_model]

        # RUN DECODER
        dec_output = self.decoder(
            input=latent_q_patched_emb_shifted,
            encoder_output=encoded_patched,
        )  # [B', seq_length_dec, hidden_dim]

        mean = self.mean_head(dec_output)  # [B', seq_length_dec, hidden_dim]
        scale = self.scale_head(dec_output)  # [B', seq_length_dec, hidden_dim]

        return mean, scale, dec_output

    def _get_encoded_seqs(
        self, previous_latents: Sequence[PreviousLatent]
    ) -> List[Tensor]:
        """
        Extract the previously procesed latents, repeating them if the
        number of processed latents is less than the context legnth

        Args:
            previous_latents: sequence of sizse at most `context_len`, containing object
                of type PreviousLatent, with two attributes:
                    - `processed` tensor of shape [b', seq_len_enc, d_model]
                    - `quantized` tensor of shape [B, C, H, W], NOT needed in this method

        Returns:
            List of length `context_len` with tensors of shape [b', seq_len_enc, d_model],
                containing `processed` data only (encoder processed data).
        """
        encoded_seqs = [p.processed for p in previous_latents]
        if len(encoded_seqs) < self.context_len:
            if self.context_len == 2:
                # encoded_seqs is a list of size 1
                return encoded_seqs * 2  # [b', seq_len_enc, d_model]*2
            elif self.context_len == 3:
                return (
                    encoded_seqs * 3  # [b', seq_len_enc, d_model]*3
                    if len(encoded_seqs) == 1
                    # repeat the 0th twice
                    else [encoded_seqs[0]] * 2 + [encoded_seqs[1]]
                )
            else:
                ValueError(f"Unsupported context_len={self.context_len}")
        return encoded_seqs

    def forward(
        self,
        latent_unquantized: Tensor,
        previous_latents: Sequence[PreviousLatent],
    ) -> TemporalEntropyModelOut:
        """
        Args:
            latent_unquantized: the latent to transmit (quantize), expected shape is
                [B, C, H, W]
            previous_latents: previously transmitted (quantized) latents, should be of
                size at least one and at most `context_len`. Each PreviousLatent has
                    - quantized: floats (i.e. noised) tensor of shape [B, C, H, W]
                    - processed: [b', seq_len_enc, d_model]

        Returns:
            TemporalEntropyModelOut, see docstring there.
        """
        H, W = latent_unquantized.shape[-2:]
        # encoded_seqs: list of tensors [b', seq_len_enc=patch_size_enc^2, d_model]
        encoded_seqs = self._get_encoded_seqs(previous_latents=previous_latents)
        b_enc, _, d_enc = encoded_seqs[0].shape  # [b', patch_size_enc^2, d_enc]
        if d_enc != self.d_model:
            raise ValueError(f"Shape mismatch, {d_enc}!={self.d_model}")

        # Soft rounding via straight-through gradient estimation
        latent_q = self.round_st(latent_unquantized)  # [B, C, H, W], float32 of ints

        # Patcher expects [B, C, H, W] and returns [b', seq_len, C]
        latent_q_patched, (n_h, n_w) = self.patcher(latent_q, self.window_size_dec)
        b_dec, seq_len, d_dec = latent_q_patched.shape  # b_dec, patch_size_dec^2, C
        if d_dec != self.num_channels:
            raise ValueError(f"Model dims don't match, {d_dec}!={self.num_channels}")
        if b_dec != b_enc:
            raise ValueError(f"Batch dims don't match, got {b_enc} != {b_dec}!")
        assert seq_len == self.window_size_dec**2, "Error patching"

        # Transformer expects inputs to have channels in the last dim, [b', seq_len, C]
        # Transformer expects quantized latents
        mean, scale, dec_output = self._get_transformer_output(
            encoded_patched=torch.cat(encoded_seqs, dim=-2),  # cat on seq dim
            latent_q_patched=latent_q_patched,  # [b_dec, seq_len, C], ROUNDED
        )
        assert (
            mean.shape == latent_q_patched.shape
        ), f"Shape mismatch! {mean.shape} != {latent_q_patched.shape}"

        decoder_features = self.patcher.unpatch(
            x_patched=(dec_output, (n_h, n_w)), crop=(H, W), channels_last=True
        )  # [B, d_model, H, W]

        latent_unquantized_patched = self.patcher(
            latent_unquantized, self.window_size_dec
        ).tensor  # [b', seq_len_dec, C]

        assert latent_unquantized_patched.shape == scale.shape  # [B', seq_len_dec, C]
        # We use GaussianConditional from compressai as bottleneck, its forward method
        # returns (output, likelihood)
        # in VCT scales and means are quantized in the bottleneck
        output, likelihood = self.bottleneck(
            inputs=latent_unquantized_patched, scales=scale, means=mean
        )  # output is a noised version of the `inputs`
        assert output.shape == likelihood.shape  # [b', seq_len, C]

        output = self.patcher.unpatch(
            (output, (n_h, n_w)), crop=(H, W), channels_last=True
        )
        assert output.shape == latent_unquantized.shape

        return TemporalEntropyModelOut(
            perturbed_latent=output, bits=likelihood, features=decoder_features
        )

    def _get_mean_scale_jitted(
        self, *, encoded_patched: Tensor, latent_q_patched: Tensor
    ):
        """
        TODO implement JIT version of the transformer forward pass
            (mean, scale, dec_output = self._get_transformer_output(...))

        Args:
            encoded_patched: what we condition on, tensor of shape
                [b', context_len*seq_len_enc, d_model]
            latent_q_patched: tensor of shape [b', seq_len_dec, C]

        Returns:
            runs the transformer and returns mean, scale and decoder_output
        """
        raise NotImplementedError("JIT not supported yet.")

    def validate_causal(self, latent_q_patched: Tensor, encoded: Tensor) -> None:
        """
        Validate that the masking is causal
        """
        # run model
        masked_means, masked_scales, _ = self._get_transformer_output(
            encoded_patched=encoded, latent_q_patched=latent_q_patched
        )
        # run model iteratively
        current_inp = torch.full_like(latent_q_patched, fill_value=10.0)
        autoreg_means = torch.full_like(latent_q_patched, fill_value=10.0)
        autoreg_scales = torch.full_like(latent_q_patched, fill_value=10.0)

        for i in range(self.seq_len_dec):
            if i > 0:  # first token is the learnt StartSym
                current_inp[:, i - 1, :] = latent_q_patched[:, i - 1, :]
            mean_i, scale_i, _ = self._get_transformer_output(
                encoded_patched=encoded, latent_q_patched=current_inp
            )
            autoreg_means[:, i, :] = mean_i[:, i, :]
            autoreg_scales[:, i, :] = scale_i[:, i, :]

        isclose_means = autoreg_means.isclose(masked_means).all()
        isclose_scales = autoreg_scales.isclose(masked_scales).all()
        causal = isclose_means and isclose_scales
        if not causal:
            msg_mean = "" if isclose_means else "means"
            msg_scales = "" if isclose_scales else "scales"
            raise ValueError(
                f"Larger than expected discrepancy: {msg_mean} {msg_scales}"
            )

    def compress(
        self,
        *,
        latent_unquantized: Tensor,
        previous_latents: Sequence[PreviousLatent],
        run_decode: bool = False,
        validate_causal: bool = False,
    ) -> TemporalEntropyModelOut:
        """
        Compress and decompress autoregressively. Can only handle batch size 1.

        Args:
            latent_unquantized: unquantized latent of shape [1, C, H, W]
            previous_latents: a sequence of length at least 1 and at most `context_len`
                containingg PreviousLatent objects which hold 2 tensors:
                    - quantized: [1, C, H, W]
                    - processed: latents passed through the encoder, expected shape
                        [b', seq_len_enc, d_model]
            run_decode: bool, defaults to False. Whether to run the actual decoding

        Returns:
            TemporalEntropyModelOut object with the following components:
                - perturbed_latent: tensor of ints with same shape as the input tensor
                    `latent_unquatnized` -- [1, C, H, W]
                - bits: number of bits used to compress the input latent, float tensor
                - features: features tensor from the decoder, shape [1, d_model, H, W]
        """
        B, C, H, W = latent_unquantized.shape
        assert B == 1, "Cannot handle batch yet."

        encoded_seqs = self._get_encoded_seqs(previous_latents)
        # previously coded latents, shape is [b', 2*seq_len_enc, d_model], floats
        encoded = torch.cat(encoded_seqs, -2)

        latent_patched, (n_h, n_w) = self.patcher(
            latent_unquantized, self.window_size_dec
        )  # [b', seq_len, C]

        if validate_causal:
            self.validate_causal(latent_q_patched=latent_patched, encoded=encoded)

        # Encoding: compress to strings - strings is a list of len seq_len_dec (16)
        strings, extra = self._encode(latent_patched, encoded)
        means, scales, dec_output, quantized = extra.values()

        decoder_features = self.patcher.unpatch(
            (dec_output, (n_h, n_w)), crop=(H, W), channels_last=True
        )  # [1, d_model, H, W]

        # Count bits in each sequence, each string is a list of len 1
        bits = [sum(len(string[0]) * 8 for string in strings)]

        # decoding:
        if not run_decode:
            # For performance, since coding is lossless, real decode can be skipped
            decoded = torch.round(latent_unquantized)
        else:
            use_output_from_encode = True
            decoded = self._decode(
                strings,
                encoded,
                shape=(H, W, C),
                encode_means=means,
                encode_scales=scales,
                use_output_from_encode=use_output_from_encode,
            )
            dequantized = self.patcher.unpatch(
                x_patched=(quantized, (n_h, n_w)),
                crop=(H, W),
                channels_last=True,
            )
            if use_output_from_encode:
                # This should pass if `use_output_from_encode=True`
                assert (decoded == dequantized).all(), "Something went wrong!"

        return TemporalEntropyModelOut(
            perturbed_latent=decoded,
            bits=torch.tensor(bits, dtype=torch.float32),
            features=decoder_features,
        )

    def _encode(
        self, latent_patched: Tensor, encoded: Tensor
    ) -> Tuple[List[str], Dict[str, Tensor]]:
        """
        Compress patched latents to strings

        Args:
            latent_patched: unquantized latent of shape [b', seq_len_dec, C], where
                b' = #patches * actual batch size, which is 1 (for compress/decompress)
            encoded: the "features" used to code the latent, ie what we condition on
                to predict the distribution of the current latent. This should be a
                tensor of shape [b', context_len*seq_len_enc, d_model] with b'=

        Returns:
            strings (list of strings), extra (dict with tensors).
                NB: In theory, nothing in the  dict (e.g. means and scales) should be
                be used at decode time. In practice, the below ._decode method
                uses them, to avoid issues with non-determinism of the transformer.
        """

        strings = []
        # all 3 tensors are [b', seq_len_dec, C]
        quantized = torch.full_like(latent_patched, fill_value=10.0)
        autoreg_means = torch.full_like(latent_patched, fill_value=100.0)
        autoreg_scales = torch.full_like(latent_patched, fill_value=100.0)
        # the decoder output has shape [b', seq_len_dec, d_model]
        dec_output_shape = (*latent_patched.shape[:-1], self.d_model)
        dec_output = torch.full(
            dec_output_shape,
            fill_value=100.0,
            dtype=torch.float32,
            device=quantized.device,
        )

        prev_mean = None
        prev_scale = None

        # add 0 at the end to to code the very last symbol and then break
        for i in itertools.chain(range(self.seq_len_dec), [0]):
            if prev_mean is not None and prev_scale is not None:  # ensures i > 0
                latent_i = latent_patched[:, i - 1, :]
                # # !!! unsqueeze -- batch size 1
                quantized_i, string = self.bottleneck.compress(
                    inputs=latent_i.unsqueeze(0),
                    scales=prev_scale.unsqueeze(0),
                    means=prev_mean.unsqueeze(0),
                )  # [1, b', C] tensor of ints, [string] list of a string
                strings.append(string)  # each string is a list of len 1
                # quantized should contain the mean.
                quantized[:, i - 1, :] = quantized_i.squeeze(0)

                if i == 0:
                    break

            mean_i, scale_i, dec_output_i = self._get_transformer_output(
                encoded_patched=encoded, latent_q_patched=quantized
            )  # [b', seq_len_dec, C]*2, [b', seq_len_dec, d_model]

            prev_mean = autoreg_means[:, i, :] = mean_i[:, i, :]  # [b', C]
            prev_scale = autoreg_scales[:, i, :] = scale_i[:, i, :]  # [b', C]
            dec_output[:, i, :] = dec_output_i[:, i, :]  # assigning [b', d_model]

        # NOTE: `autoreg_means` and `autoreg_scales` must not be used at decode time.
        # However, due to transofmer non-determinism, we return them and allow "fake"
        # decoding by setting use_output_from_encode=True in `._decode`
        extra = {
            "means": autoreg_means,
            "scales": autoreg_scales,
            "dec_output": dec_output,
            "quantized": quantized,
        }
        return strings, extra

    def _decode(
        self,
        strings: List[str],
        encoded: Tensor,
        shape: Sequence[int],
        encode_means: Tensor,
        encode_scales: Tensor,
        use_output_from_encode: bool = True,
    ) -> Tensor:
        """
        Decompress strings

        Args:
            strings: list of strings, length should be seq_len_dec
            encoded: previous latents, passed to transformer
            shape: H, W, C
            encode_means: means from encode (compresss) step
            encode_scales: scales from encode (compresss) step
            use_output_from_encode: use encode_means and encode_scales from the encode
                step (not real compression), or use the transformer to compute them.
                Note there could be a discrepancy between the two due to the
                non-deterministic nature of transformers (which makes them suboptimal
                for pmf prediction and use in compression), so for research purposes it
                could be acceptable to use the output from the encoder.

        Returns:
            A tenosor of the decoded strings
        """
        H, W, C = shape
        _device = encode_means.device
        fake_patched = self.patcher(
            torch.ones((1, C, H, W), device=_device), self.window_size_dec
        )  # placeholder object Patched(tensor, num_patches)
        decompressed = torch.full_like(
            fake_patched.tensor, fill_value=10.0
        )  # [b', seq_len_dec, C]

        prev_mean = None
        prev_scale = None
        for i in itertools.chain(range(self.seq_len_dec), [0]):
            if prev_mean is not None and prev_scale is not None:
                decoded_i = self.bottleneck.decompress(
                    strings=strings.pop(0),
                    scales=prev_scale.unsqueeze(0),
                    means=prev_mean.unsqueeze(0),  #
                )  # decompressed i-th token, [b', C]
                decompressed[:, i - 1, :] = decoded_i
                if i == 0:
                    break
            # predict mean and scale for the i-th token, given previously decompressed
            mean_i, scale_i, _ = self._get_transformer_output(
                encoded_patched=encoded, latent_q_patched=decompressed
            )  # [b', seq_len_dec, C]x2

            target_mean, target_scale = encode_means[:, i, :], encode_scales[:, i, :]
            actual_mean, actual_scale = mean_i[:, i, :], scale_i[:, i, :]

            if use_output_from_encode:
                # NOTE: To deal with non-deterministm of the transformer, use the means
                # and the scales from encoding, and log errors. Note that this cannot
                # be done in practice.
                prev_mean = target_mean  # mean of current token, [b', C]
                prev_scale = target_scale  # scale of current token, [b', C]
                error_mean = (actual_mean - target_mean).abs().sum()
                error_scale = (actual_scale - target_scale).abs().sum()
                percent_of_total_mean = error_mean / target_mean.abs().sum()
                percent_of_total_scale = error_scale / target_scale.abs().sum()
                if percent_of_total_mean > 0.01 or percent_of_total_scale > 0.01:
                    print(
                        "Larger than expected discrepancy in transformer output found! ",
                        f"Decode step {i}: mean error = {100*percent_of_total_mean}%, ",
                        f"mean error = {100*percent_of_total_scale}%",
                        file=sys.stderr,
                    )
            else:
                prev_mean = actual_mean  # mean of current token, [b', C]
                prev_scale = actual_scale  # scale of current token, [b', C]

        assert not strings
        return self.patcher.unpatch(
            x_patched=(decompressed, fake_patched.num_patches),
            crop=(H, W),
            channels_last=True,
        )

    def update(self, force: bool = False) -> bool:
        """
        Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later compress
        and decompress with an actual entropy coder.

        Args:
            force: overwrite previous values (default: False)

        Returns:
            updated: True if one of the bottlenecks was updated.
        """
        check = getattr(self.bottleneck, "update", None)
        if check is not None:
            bottleneck_updated = self.bottleneck.update(force=force)
        else:
            bottleneck_updated = False

        return bottleneck_updated
