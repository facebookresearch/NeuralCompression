# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, NamedTuple, Optional, Tuple, Union

from torch import Tensor


class HyperpriorOutput(NamedTuple):
    image: Tensor
    latent: Tensor
    latent_likelihoods: Tensor
    quantized_latent_likelihoods: Tensor
    hyper_latent: Tensor
    hyper_latent_likelihoods: Tensor
    quantized_hyper_latent_likelihoods: Tensor
    quantized_latent: Optional[Tensor] = None
    quantized_hyper_latent: Optional[Tensor] = None


class HyperpriorCompressedOutput(NamedTuple):
    latent_strings: Union[List[str], List[List[str]]]
    hyper_latent_strings: List[str]
    image_size: Tuple[int, int]
    padded_size: Tuple[int, int]
