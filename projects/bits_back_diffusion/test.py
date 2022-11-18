"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
import numpy.random as rng
import torch
from bits_back_diffusion.codec import (
    BitsBackCodec,
    DiffusionModel,
    GaussianParams,
    diag_gaussian_unif_bins,
)
from bits_back_diffusion.script_util import create_gaussian_diffusion
from craystack import rans
from craystack.codecs_test import check_codec


def test_gaussian_ub(
    data_min: Tuple[float, float] = (-2.0, -3.0),
    data_max: Tuple[float, float] = (2.0, 3.0),
    bin_prec: int = 10,
    coding_prec: int = 16,
    batch_size: int = 5,
):
    """
    Test the codec for the diagonal Gaussian with uniform bins.

    Based on `craystack.codecs_test.test_gaussian_ub`
    (https://github.com/j-towns/craystack).
    """
    mean = rng.randn(batch_size, 2) / 10
    stdd = np.exp(rng.random((batch_size, 2)) / 2)

    data = np.array([rng.choice(1 << bin_prec, 2) for _ in range(batch_size)])
    check_codec(
        (batch_size, 2),
        diag_gaussian_unif_bins(
            mean, stdd, np.array(data_min), np.array(data_max), coding_prec, bin_prec
        ),
        data,
    )


class CheatingDiffusionModel(DiffusionModel):
    """
    Diffusion model which uses the true posterior instead of the prior.
    """

    def __init__(self, data: np.ndarray, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = data

    def generate_fn(
        self, latent: np.ndarray, step: int, data: Optional[np.ndarray] = None
    ) -> GaussianParams:
        return super().generate_fn(latent=latent, step=step, data=self.data)


def test_codec(
    batch_size: int = 2,
    image_size: int = 3,
    data_prec: int = 8,
    data_min: float = -1,
    data_max: float = 1,
    steps: int = 40,
):
    """
    Test the Bits-Back codec for diffusion models.
    """
    data_shape = (batch_size, 3, image_size, image_size)
    diffusion = create_gaussian_diffusion(timestep_respacing=f"vlb{steps}")
    discr_data = rng.randint(low=0, high=(1 << data_prec) - 1, size=data_shape)
    data = data_min + discr_data.astype(float) * (data_max - data_min) / (
        (1 << data_prec) - 1
    )
    codec_model = CheatingDiffusionModel(
        data, diffusion=diffusion, model=torch.nn.Linear(1, 1)
    )
    codec = BitsBackCodec(
        model=codec_model,
        data_shape=data_shape,
        data_min=data_min,
        data_max=data_max,
        track=True,
    )
    message = deepcopy(codec.message)
    codec.encode(discr_data)
    stats = codec.statistics()
    rate_extra = codec._bytes_to_bpd(codec.extra_bytes)
    _, decoded_data = codec.decode()

    assert stats.rate_effective_push is not None
    assert stats.rate_effective_pop is not None
    np.testing.assert_equal(discr_data, decoded_data)
    np.testing.assert_allclose(stats.rate_effective_push, stats.rate_effective_pop)

    assert rans.message_equal(message, codec.message)
    assert codec.flat_message.nbytes == codec.initial_bytes
    assert stats.samples == batch_size
    assert np.isclose(stats.rate_effective, stats.rate_effective_push[0])
    assert np.isclose(stats.rate_effective, 0.0)
    assert np.isclose(stats.rate_used, rate_extra)
    assert np.isclose(stats.rate_used, sum(stats.rate_effective_push))
    assert codec.data_count == 0
