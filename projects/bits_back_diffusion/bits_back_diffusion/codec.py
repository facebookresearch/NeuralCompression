"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from abc import ABC, abstractmethod
from typing import List, NamedTuple, Optional, Tuple, Union, overload

import craystack as cs
import numpy as np
import torch
from craystack import codecs
from craystack.bb_ans import BBANS
from improved_diffusion.gaussian_diffusion import GaussianDiffusion

GaussianParams = NamedTuple(
    "GaussianParams", [("mean", np.ndarray), ("stdd", np.ndarray)]
)


DiscrLatent = NamedTuple(
    "DiscrLatent", [("inds", np.ndarray), ("params", GaussianParams)]
)


CodecStats = NamedTuple(
    "CodecStats",
    [
        ("samples", int),
        ("rate_total", float),
        ("rate_used", float),
        ("rate_effective", float),
        ("rate_effective_push", Optional[np.ndarray]),
        ("rate_effective_pop", Optional[np.ndarray]),
    ],
)


class BaseHLVModel(ABC):
    """
    Base class for hierarchical latent variable models.

    We assume that the model is of the form

    z_{steps-1} -> z_{steps-2} -> ... -> z_0 -> data,

    where

    1) all variables have the same dims [batch-size, channels, height, width],
    2) p(z_{step}|z_{step+1}), q(z_{step}|z_{step+1}, data), q(z_{steps-1}|data)
        are Gaussians with diagonal covariance matrix,
    3) p(z_{steps-1}) is a standard normal.

    Args:
        steps: The number of steps/hierarchchies.
    """

    def __init__(self, steps: int):
        self.steps = steps

    @abstractmethod
    def infer_fn(self, data: np.ndarray) -> GaussianParams:
        """
        Outputs the mean and standard deviation of the Gaussian
        posterior q(z_{steps-1}|data).

        Args:
            data: The input data array.
        Returns:
            The parameters (mean and stdd) of the Gaussian posterior.
        """

    @abstractmethod
    def generate_fn(
        self, latent: np.ndarray, step: int, data: Optional[np.ndarray] = None
    ) -> GaussianParams:
        """
        If `data` is None, outputs the mean and stdd of the Gaussian
        prior p(z_{step}|z_{step+1}), where latent equals z_{step+1}.
        If `data` is provided, uses the posterior q(z_{step}|z_{step+1}, data).

        Args:
            latent: The latent z_{step+1} at the given step.
            step: The number of steps (minus 1). Here, 0 means one step.
            data: The input data array.
        Return:
            The parameters (mean and standard deviation)
            of the Gaussian prior or posterior.
        """


class DiffusionModel(BaseHLVModel):
    """
    Hierarchical latent variable model based on a diffusion model.

    Args:
        model: The model to use for the denoising (should be in eval mode).
        diffusion: The diffusion to sample from.
        use_log_variance: Whether to use the (clipped) log variance.
        clip_denoised: If True, clip the denoised signal.
        device: The device to use.
            Defaults to the device of the first model parameter.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        diffusion: GaussianDiffusion,
        use_log_variance: bool = False,
        clip_denoised: bool = True,
        device: Union[torch.device, str, None] = None,
    ):
        self.model = model
        self.diffusion = diffusion
        self.use_log_variance = use_log_variance
        self.clip_denoised = clip_denoised
        self.device = device
        if self.device is None:
            self.device = next(model.parameters()).device
        super().__init__(steps=self.diffusion.num_timesteps)

    def _gaussian_params(
        self, mean: torch.Tensor, variance: torch.Tensor, log_variance: torch.Tensor
    ) -> GaussianParams:
        if self.use_log_variance:
            stdd = torch.exp(0.5 * log_variance)
        else:
            stdd = torch.sqrt(variance)
        return GaussianParams(
            mean=mean.cpu().numpy(),
            stdd=stdd.cpu().numpy(),
        )

    def infer_fn(self, data: np.ndarray) -> GaussianParams:
        """
        Outputs the mean and standard deviation of the Gaussian
        posterior q(z_{step}|data).

        Args:
            data: The input data array.
        Returns:
            The parameters (mean and stdd) of the Gaussian posterior.
        """
        t = torch.tensor([self.steps - 1], dtype=torch.long).expand(data.shape[0])
        x_start = torch.from_numpy(data).to(dtype=torch.float)

        mean, variance, log_variance = self.diffusion.q_mean_variance(
            x_start,
            t=t,
        )
        return self._gaussian_params(mean, variance, log_variance)

    def generate_fn(
        self, latent: np.ndarray, step: int, data: Optional[np.ndarray] = None
    ) -> GaussianParams:
        """
        If `data` is None, outputs the mean and stdd of the Gaussian
        prior p(z_{step}|z_{step+1}), where latent equals z_{step+1}.
        If `data` is provided, uses the posterior q(z_{step}|z_{step+1}, data).

        Args:
            latent: The latent z_{step+1} at the given step.
            step: The number of diffusion steps (minus 1).
                Here, 0 means one step.
            data: The input data array.
        Returns:
            The parameters (mean and standard deviation)
            of the Gaussian prior or posterior.
        """
        t = torch.tensor([step], dtype=torch.long).expand(latent.shape[0])
        x_t = torch.from_numpy(latent).to(dtype=torch.float)

        if data is None:
            with torch.no_grad():
                output = self.diffusion.p_mean_variance(
                    model=self.model,
                    x=x_t.to(self.device),
                    t=t.to(self.device),
                    clip_denoised=self.clip_denoised,
                )
            mean, variance, log_variance = (
                output["mean"],
                output["variance"],
                output["log_variance"],
            )
        else:
            x_start = torch.from_numpy(data).to(dtype=torch.float)
            mean, variance, log_variance = self.diffusion.q_posterior_mean_variance(
                x_start=x_start, x_t=x_t, t=t
            )
        return self._gaussian_params(mean, variance, log_variance)


class BitsBackCodec:
    """
    Lossless compression for Herarchical latent variable models.

    Based on the Craystack (https://github.com/j-towns/craystack)
    implementation of Bits-Back coding.

    Args:
        model: The hierarchical latent variable model.
        data_shape: The shape [batch-size, channels, height, width]
            of the data and each latent.
        latent_prec: Coding precision.
        prior_prec: Precision for the discretization.
        obs_prec: Precision for the observation codec.
        data_prec: Precision of the discrete data.
        initial_len: Length of the random initial bitstring.
        data_min: Minimum value of the data to be fed into the model.
        data_max: Maximum value of the data to be fed into the model.
        clip_stdd: Minimum value for the stdd of the observation codec.
        track: Whether to track the message sizes at each step.
    """

    def __init__(
        self,
        model: BaseHLVModel,
        data_shape: Tuple[int, int, int, int],
        latent_prec: int = 18,
        prior_prec: int = 10,
        obs_prec: int = 24,
        data_prec: int = 8,
        initial_len: Optional[int] = None,
        data_min: float = -1.0,
        data_max: float = 1.0,
        clip_stdd: float = 1e-6,
        track: bool = False,
    ):
        # model and diffusion
        self.model = model
        if not len(data_shape) == 4:
            raise ValueError(
                f"`data_shape` is {data_shape} but should be [b, c, h, w]."
            )
        self.data_shape = data_shape
        self.data_size = np.prod(data_shape).item()
        self.data_min = data_min
        self.data_max = data_max
        self.clip_stdd = clip_stdd

        # codec
        self.latent_prec = latent_prec
        self.prior_prec = prior_prec
        self.obs_prec = obs_prec
        self.data_prec = data_prec
        self.track = track
        self.prior_codec = cs.Uniform(self.prior_prec)

        # initialize
        self.extra_bytes = 0
        self.data_count = 0
        self.push_bytes = np.zeros(self.model.steps + 1, dtype=np.uint64)
        self.pop_bytes = np.zeros(self.model.steps + 1, dtype=np.uint64)
        if initial_len is None:
            # sensible initial value to prevent popping from an empty message
            initial_len = (
                self.data_size
                * self.model.steps
                * max(self.prior_prec, self.latent_prec)
                // 32
            )

        self.message = cs.random_message(initial_len, self.data_size)
        self.initial_bytes = self.flat_message.nbytes

    @property
    def bb_ans(self) -> cs.Codec:
        """
        Craystack codec with correct view.

        Returns:
            The craystack Bits-Back codec.
        """
        codec = BBANS(
            cs.Codec(self.prior_push, self.prior_pop),
            self.likelihood,
            self.posterior,
        )
        return cs.substack(codec, self._model_view)

    @property
    def flat_message(self) -> np.ndarray:
        """
        Flatten a message head and tail into a 1d array. This maps to a
        message representation which can be used to measure the message length
        and which can easily be saved to disk.

        Returns:
            The 1d numpy array corresponding to the craystack message.
        """
        return cs.flatten(self.message)

    def _model_view(self, head: np.ndarray) -> np.ndarray:
        return np.reshape(head[: self.data_size], self.data_shape)

    @overload
    def _bytes_to_bpd(self, nbytes: int) -> float:
        ...

    @overload
    def _bytes_to_bpd(self, nbytes: np.ndarray) -> np.ndarray:
        ...

    def _bytes_to_bpd(self, nbytes: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute the bits per dimension from the bytes of message.

        Args:
            nbytes: Bytes of a message or array of bytes.
        Returns:
            The bits per dimension.
        """
        return 8 * nbytes / (self.data_count * self.data_size)

    def _to_latent(self, discr_latent: DiscrLatent) -> np.ndarray:
        """
        Compute the latent from indices corresponding to bins
        having equal mass under Gaussian priors.

        Args:
            discr_latent: The discretized latent specifying the indices and
                the parameters (mean and stdd) of the Gaussian prior.
        Returns:
            The bins corresponding to the latent.
        """
        params = discr_latent.params
        return (
            params.mean
            + cs.std_gaussian_centres(self.prior_prec)[discr_latent.inds] * params.stdd
        )

    def _post_codec(
        self, post_params: GaussianParams, prior_params: GaussianParams
    ) -> cs.Codec:
        """
        Craystack codec for the posterior.

        Args:
            post_params: The parameters (mean and standard deviation)
                of the Gaussian posterior.
            prior_params: The parameters (mean and standard deviation)
                of the Gaussian prior.
        Returns:
            The craystack codec for the Gaussian posterior
            (discretized according to the Gaussian prior).
        """
        return cs.DiagGaussian_GaussianBins(
            post_params.mean,
            post_params.stdd,
            prior_params.mean,
            prior_params.stdd,
            self.latent_prec,
            self.prior_prec,
        )

    def prior_push(self, message: tuple, discr_latents: List[DiscrLatent]) -> tuple:
        """
        Push indices idx_1, ..., idx_{steps-1} bottom-up
        using a uniform distribution.

        Args:
            message: The craystack message to push to.
            discr_latents: The list of discretized latents
                to be pushed to the message.
        Returns:
            The new craystack message with the indices appended.
        """
        assert len(discr_latents) == self.model.steps
        for step in range(1, self.model.steps + 1):
            codec = self._track_codec(self.prior_codec, step=step)
            (message,) = codec.push(message, discr_latents[step - 1].inds)
        return (message,)

    def prior_pop(self, message: tuple) -> Tuple[tuple, List[DiscrLatent]]:
        """
        Pop indices idx_{steps-1}, ..., idx_1 top-down
        using a uniform distribution.

        Indices idx_{steps} correspond to bins having equal mass under the
        Gaussian priors p(z_{step}|z_{steps+1}) or p(z_{steps-1}) and
        (discretized) latent variables z_{step} are given by the median
        of the corresponding bin.

        Args:
            message: The craystack message to pop from.
        Returns:
            A tuple containing the new craystack message and the popped indices.
        """
        discr_latents: List[DiscrLatent] = []
        for step in range(self.model.steps, 0, -1):
            if step == self.model.steps:
                prior_params = GaussianParams(
                    mean=np.zeros(self.data_shape), stdd=np.ones(self.data_shape)
                )
            else:
                latent = self._to_latent(discr_latents[-1])
                prior_params = self.model.generate_fn(latent, step)

            codec = self._track_codec(self.prior_codec, step=step)
            message, latent_inds = codec.pop(message)
            discr_latents.append(DiscrLatent(inds=latent_inds, params=prior_params))
        return message, discr_latents[::-1]

    def posterior(self, discr_data: np.ndarray) -> cs.Codec:
        """
        Posterior codec for BB-ANS.

        Args:
            discr_data: The discrete data in [0, 2 ** self.data_prec - 1],
                on which the posterior is conditioned.
        Returns:
            The craystack codec corresponding to the posterior
            of the Bits-Back codec.
        """
        data = self.data_min + discr_data.astype(float) * (
            self.data_max - self.data_min
        ) / ((1 << self.data_prec) - 1)
        top_post_params = self.model.infer_fn(data)

        def posterior_push(message: tuple, discr_latents: List[DiscrLatent]) -> tuple:
            """
            Push indices idx_1, ..., idx_{steps-1} corresponding to the
            latent variables z_1, ..., z_{steps-1} using the Gaussian posteriors
            q(z_{step}|z_{step+1}, x) or q(z_{steps-1}|discr_data) and
            discretization given by bins that have equal mass under Gaussian
            priors with mean/stdd given by the parameters of the latents.

            Args:
                message: The craystack message to push to.
                discr_latents: The list of (discretized) latents
                    to be pushed to the message.
            Returns:
                The new craystack message with the indices appended.
            """
            assert len(discr_latents) == self.model.steps
            for step in range(1, self.model.steps + 1):
                # first get the posterior params
                if step < self.model.steps:
                    next_latent = self._to_latent(discr_latents[step])
                    post_params = self.model.generate_fn(
                        next_latent, step=step, data=data
                    )
                else:
                    post_params = top_post_params

                discr_latent = discr_latents[step - 1]
                codec = self._track_codec(
                    self._post_codec(post_params, discr_latent.params), step=step
                )
                (message,) = codec.push(message, discr_latent.inds)
            return (message,)

        def posterior_pop(message: tuple) -> Tuple[tuple, List[DiscrLatent]]:
            """
            Pop indices idx_1, ..., idx_{steps-1} corresponding to the
            latent variables z_1, ..., z_{steps-1} using the Gaussian posteriors
            q(z_{step}|z_{step+1}, discr_data) or q(z_{steps-1}|discr_data)
            and discretization given by bins that have equal mass
            under Gaussian priors p(z_{step}|z_{step+1}) or p(z_{steps-1}).

            Args:
                message: The craystack message to pop from.
            Returns:
                A tuple containing the new craystack message
                and the popped indices.
            """
            discr_latents: List[DiscrLatent] = []
            for step in range(self.model.steps, 0, -1):
                if step == self.model.steps:
                    prior_params = GaussianParams(
                        mean=np.zeros(self.data_shape), stdd=np.ones(self.data_shape)
                    )
                    post_params = top_post_params
                else:
                    latent = self._to_latent(discr_latents[-1])
                    prior_params = self.model.generate_fn(latent, step=step)
                    post_params = self.model.generate_fn(latent, step=step, data=data)

                codec = self._post_codec(post_params, prior_params)
                message, latent_inds = self._track_codec(codec, step=step).pop(message)
                discr_latents.append(DiscrLatent(inds=latent_inds, params=prior_params))

            if self.data_count == 0:
                # compute extra bytes needed for encoding
                self.extra_bytes = self.initial_bytes - cs.flatten(message).nbytes

            return message, discr_latents[::-1]

        return cs.Codec(posterior_push, posterior_pop)

    def likelihood(self, discr_latents: List[DiscrLatent]) -> cs.Codec:
        """
        Likelihood codec for BB-ANS.

        Extracts indices idx_0 of z_0 and mean/standard deviation of
        the prior p(z_0|z_1). Indices correspond to bins with equal
        mass under the Gaussian prior p(z_0|z_1), allowing
        to obtain the (discretized) latent variable z_0.
        The codec then uses a uniform discretization under a Gaussian with
        mean/(clipped) standard deviation of p(x|z_0) to encode/decode
        the data in [0, 2 ** self.data_prec - 1].

        Args:
            discr_latents: The list of (discretized) latents.
        Returns:
            The craystack codec corresponding to the likelihood
            of the Bits-Back codec.
        """
        assert len(discr_latents) == self.model.steps
        latent = self._to_latent(discr_latents[0])
        params = self.model.generate_fn(latent, step=0)

        codec = diag_gaussian_unif_bins(
            mean=params.mean,
            stdd=params.stdd.clip(min=self.clip_stdd),
            data_min=self.data_min,
            data_max=self.data_max,
            coding_prec=self.obs_prec,
            bin_prec=self.data_prec,
        )
        return self._track_codec(codec, step=0)

    def _track_codec(self, codec: cs.Codec, step: int) -> cs.Codec:
        """
        Track the message length at a given time-step.

        Args:
            codec: Codec that adheres to the craystack api.
            step: The step in the hierarchy.
        Returns:
            A craystack codec which tracks the pops and pushes.
        """
        if not self.track:
            return codec

        if not 0 <= step <= self.model.steps:
            raise RuntimeError(f"Step {step} is out of bounds [0, {self.model.steps}].")

        def track_push(message, data):
            init_bytes = cs.flatten(message).nbytes
            (message,) = codec.push(message, data)
            self.push_bytes[step] += cs.flatten(message).nbytes - init_bytes
            return (message,)

        def track_pop(message):
            init_bytes = cs.flatten(message).nbytes
            message, data = codec.pop(message)
            self.pop_bytes[step] += init_bytes - cs.flatten(message).nbytes
            return message, data

        return cs.Codec(track_push, track_pop)

    def quantize(self, data: np.ndarray) -> np.ndarray:
        """
        Quantize the data for coding.

        Args:
            data: Array with floating point values
                in [self.data_min, self.data_max].
        Returns:
            Quantized array with integer values in [0, 2 ** self.data_prec - 1].
        """
        data = (
            (data - self.data_min)
            * ((1 << self.data_prec) - 1)
            / (self.data_max - self.data_min)
        )
        return data.round().astype(np.int32)

    def encode(self, discr_data: np.ndarray) -> tuple:
        """
        Push data in [0, 2 ** self.data_prec - 1] using BB-ANS, i.e.,
        1) idx_{steps-1}, ..., idx_0 = posterior(discr_data).pop()
        2) likelihood(z_0).push(discr_data)
        3) prior.push(idx_0, ..., idx_{steps-1})

        Args:
            discr_data: Discrete data in [0, 2 ** self.data_prec - 1].
        Returns:
            The craystack message with the appended data.
        """
        if (discr_data.min() < 0) or (discr_data.max() > (1 << self.data_prec) - 1):
            raise ValueError(
                "The values of `discr_data` are in "
                f"[{discr_data.min()},{discr_data.max()}] "
                "but should be in [0, 2 ** self_data_prec - 1]."
            )
        if not discr_data.shape == self.data_shape:
            raise ValueError(
                f"The shape of `discr_data` is {discr_data.shape} "
                f"but should be {self.data_shape}."
            )

        (message,) = self.bb_ans.push(self.message, discr_data)
        self.data_count += 1
        self.message = message
        return message

    def decode(self) -> Tuple[tuple, np.ndarray]:
        """
        Pop data using BB-ANS, i.e.,
        1) idx_{steps-1}, ..., idx_0 = prior.pop()
        2) discr_data = likelihood(z_0).pop()
        3) posterior(discr_data).push(idx_0, ..., idx_{steps-1})

        Returns:
            A tuple containing the new craystack message and the popped data.
        """
        message, discr_data = self.bb_ans.pop(self.message)
        self.data_count -= 1
        self.message = message
        return message, discr_data

    def statistics(self) -> CodecStats:
        """
        Output various bit-rates per dimension.

        Returns:
            A namedtuple containing the number of encoded samples and the
            total/used/effective bits per dimension used for encoding
            the samples. If `self.track`, also returns the bits per
            dimension used in the pop/push of every diffusion step.
        """
        message_bytes = self.flat_message.nbytes
        samples = self.data_count * self.data_shape[0]
        rate_total = self._bytes_to_bpd(message_bytes)
        rate_used = self._bytes_to_bpd(
            message_bytes - self.initial_bytes + self.extra_bytes
        )
        rate_effective = self._bytes_to_bpd(message_bytes - self.initial_bytes)
        rate_effective_push = (
            self._bytes_to_bpd(self.push_bytes) if self.track else None
        )
        rate_effective_pop = self._bytes_to_bpd(self.pop_bytes) if self.track else None
        return CodecStats(
            samples=samples,
            rate_total=rate_total,
            rate_used=rate_used,
            rate_effective=rate_effective,
            rate_effective_push=rate_effective_push,
            rate_effective_pop=rate_effective_pop,
        )

    def state_dict(self) -> dict:
        """
        Output the state of the codec, e.g., for saving.

        Returns:
            A dict containing the flattened message, the data count,
            and the extra/push/pop/initial bytes.
        """
        return {
            "flat_message": self.flat_message,
            "data_count": self.data_count,
            "extr_bytes": self.extra_bytes,
            "push_bytes": self.push_bytes,
            "pop_bytes": self.pop_bytes,
            "initial_bytes": self.initial_bytes,
        }

    def load_state_dict(self, state_dict: dict):
        """
        Load the state of the codec as given by `state_dict`.

        Args:
            state_dict: A dict mapping a subset of attribute names
                of the instance to their new values and/or the key
                `flat_message` to the new flattened message.
        """
        for key in state_dict:
            if key == "flat_message":
                self.message = cs.unflatten(state_dict[key], self.data_shape)
            else:
                setattr(self, key, state_dict[key])


def diag_gaussian_unif_bins(
    mean: np.ndarray,
    stdd: np.ndarray,
    data_min: Union[float, np.ndarray],
    data_max: Union[float, np.ndarray],
    coding_prec: int,
    bin_prec: int,
) -> cs.Codec:
    """
    Codec for data from a diagonal Gaussian with uniform bins.

    Based on `craystack.codecs.DiagGaussian_UnifBins`
    (https://github.com/j-towns/craystack)
    to extend first/last bins until minus/plus infinity.

    Args:
        mean: Mean of the Gaussian.
        stdd: Standard deviation of the Gaussian.
        data_min: Minimum of the data.
        data_max: Maximum of the data.
        coding_prec: Precision to use for coding.
        bin_prec: The precision to use for the uniform bins.
    Returns:
        The craystack codec for encoding Gaussian data
        discretized using uniform bins.
    """
    n_bins = 1 << bin_prec
    bin_min = data_min - 1 / (n_bins - 1)
    bin_max = data_max + 1 / (n_bins - 1)

    bins = np.linspace(bin_min, bin_max, n_bins + 1)
    bins = np.broadcast_to(np.moveaxis(bins, 0, -1), mean.shape + (n_bins + 1,))
    cdfs = codecs.norm.cdf(bins, mean[..., np.newaxis], stdd[..., np.newaxis])
    cdfs[..., 0] = 0
    cdfs[..., -1] = 1
    pmfs = cdfs[..., 1:] - cdfs[..., :-1]
    buckets = codecs._cumulative_buckets_from_probs(pmfs, coding_prec)
    enc_statfun = codecs._cdf_to_enc_statfun(
        codecs._cdf_from_cumulative_buckets(buckets)
    )
    dec_statfun = codecs._ppf_from_cumulative_buckets(buckets)

    return codecs.NonUniform(enc_statfun, dec_statfun, coding_prec)
