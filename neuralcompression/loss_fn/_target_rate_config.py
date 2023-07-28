# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple


class TargetRateConfig:
    """
    HifiC-style target rate config.

    This class wraps all variables used for rate targeting as used in the HiFiC
    paper. When the calculated rate is higher than the target, a higher coding
    cost is applied. Vice versa when the calculated rate is lower.

    This class alone does not apply the rate targeting - it is simply a
    convenience data wrapper to help the user.

    See the HiFiC paper for further details:

    High-Fidelity Generative Image Compression
    Fabian Mentzer, George D. Toderici, Michael Tschannen, Eirikur Agustsson

    Args:
        target_bpp: The target bits-per-pixel rate.
        target_factors: The scaling values for different levels of training.
            For example, a value of [1.4, 1.0] would shift the target rate
            40% higher at the beginning of training before lowering it to
            ``target_bpp``.
        target_steps: Training steps to adjust ``target_bpp`` based on
            ``target_factors``.
        lam_levels: Lambda levels for adjusting coding cost, with the first
            being smaller than the second.
        lam2_factors: Same as ``target_factors`` - adjust the second lambda
            values based on training stage.
        lam2_steps: Same as ``target_steps`` - adjust the second lambda values
            at these step intervals.
    """

    def __init__(
        self,
        target_bpp: float,
        target_factors: List[float],
        target_steps: List[int],
        lam_levels: Tuple[float, float],
        lam2_factors: List[float],
        lam2_steps: List[int],
    ):
        if len(target_steps) + 1 != len(target_factors) or len(lam2_steps) + 1 != len(
            lam2_factors
        ):
            raise ValueError(
                "Expect length of steps list to be 1 less than factors list."
            )
        self._target_bpp = target_bpp
        self._target_factors = target_factors
        self._target_steps = target_steps
        self._lam_levels = lam_levels
        self._lam2_factors = lam2_factors
        self._lam2_steps = lam2_steps

    @property
    def target_bpp(self) -> float:
        return self._target_bpp

    @property
    def target_factors(self) -> List[float]:
        return self._target_factors

    @property
    def target_steps(self) -> List[int]:
        return self._target_steps

    @property
    def lam_levels(self) -> Tuple[float, float]:
        return self._lam_levels

    @property
    def lam2_factors(self) -> List[float]:
        return self._lam2_factors

    @property
    def lam2_steps(self) -> List[int]:
        return self._lam2_steps
