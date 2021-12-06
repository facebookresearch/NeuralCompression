# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, NamedTuple, Optional, Sequence, Tuple

import jax.lax as lax
import jax.numpy as jnp
from jax.numpy import ndarray

from ._backend import CrayCompressedMessage, peek, pop_symbols, push_symbols


class CrayCodec(NamedTuple):
    """
    Codec for Craystack rANS algorithm.

    Craystack defines an API that generalizes the rANS algorithm. To define a
    codec for Craystack, the user must define the push and pop operations for
    how rANS interacts with the compressed message stack. In addition, for the
    NeuralCompression implementation, we also have the user define a,
    ``cdf_state``, which is carried through the algorithm, allowing
    context-adaptive coding or logging.

    The Craystack codec algorithm is described in `A tutorial on the range
    variant of asymmetric numeral systems (J. Townsend)
    <https://arxiv.org/abs/2001.09186>`_.

    Args:
        push: A Python closure function that defines how to push a symbol
            on to the data stack.
        pop: A Python closure function that defines how to pop a symbol
            from the data stack.
        cdf_state: An initial state for the CDF function, used for
            context-adaptive coding or logging.
        message_dtype: A string specifying the data type of the message.
        allows_empty_pops: A boolean that specifys if CrayTail should be
            allowed to create random bits on demand. See the ``pop_symbols``
            method in ``entropy_coders.craystack._backend.py``.
    """

    push: Callable[
        [ndarray, CrayCompressedMessage, Sequence[ndarray]],
        Tuple[CrayCompressedMessage, Sequence[ndarray]],
    ]
    pop: Callable[
        [ndarray, CrayCompressedMessage, ndarray, Sequence[ndarray]],
        Tuple[CrayCompressedMessage, ndarray, Sequence[ndarray]],
    ]
    cdf_state: Optional[Sequence[ndarray]] = None
    message_dtype: str = "int64"
    allow_empty_pops: bool = False


def fixed_array_cdf_codec(
    cdf_array: ndarray,
    precision: int = 16,
    message_dtype: str = "int64",
    allow_empty_pops: bool = False,
) -> CrayCodec:
    """
    Simple Craystack codec using CDF array.

    Given an ``N+1``-length array that defines a cumulative distribution
    function for ``N`` symbols, this function builds a Craystack codec by
    defining push and pop operations that can be used by the internals of the
    rANS algorithm.

    Args:
        cdf_array: An ``N+1``-length array defining the CDF for ``N`` symbols
            where the frequency count of symbol ``s`` is
            ``cdf_array[s+1] - cdf_array[s]``.
        precision: An integer specifying codec operating precision.
        message_dtype: Data type for input and output messages.
        allows_empty_pops: A boolean that specifys if CrayTail should be
            allowed to create random bits on demand. See the ``pop_symbols``
            method in ``entropy_coders.craystack._backend.py``.

    Returns:
        A Craystack codec that can be input to the coder.
    """
    init_cdf_state = (cdf_array,)

    def _lookup(cdf, symbols):
        return jnp.take_along_axis(cdf, symbols[..., None], -1)[..., 0]

    def cdf_fun(symbols, cdf_state):
        cdf = cdf_state[0]
        return _lookup(cdf, symbols), _lookup(cdf, symbols + 1), cdf_state

    def inverse_cdf_fun(cdf_value, cdf_state):
        cdf = cdf_state[0]
        symbols = jnp.argmin(jnp.expand_dims(cdf_value, -1) >= cdf, axis=-1) - 1
        return _lookup(cdf, symbols), _lookup(cdf, symbols + 1), symbols, cdf_state

    return default_rans_codec(
        cdf_fun,
        inverse_cdf_fun,
        init_cdf_state,
        precision,
        message_dtype,
        allow_empty_pops,
    )


def default_rans_codec(
    cdf_fun: Callable[
        [ndarray, Sequence[ndarray]], Tuple[ndarray, ndarray, Sequence[ndarray]]
    ],
    inverse_cdf_fun: Callable[
        [ndarray, Sequence[ndarray]],
        Tuple[ndarray, ndarray, ndarray, Sequence[ndarray]],
    ],
    init_cdf_state: Sequence[ndarray],
    precision: int = 16,
    message_dtype: str = "int64",
    allow_empty_pops: bool = False,
) -> CrayCodec:
    """
    Default rANS codec for Craystack.

    This function takes as input a functional CDF, an inverse CDF and an
    initial CDF state and converts to a Craystack codec that can be input by
    defining push and pop operations that can be used by the internals of the
    rANS algorithm.

    Args:
        cdf_fun: A ``Callable`` that maps a symbol and a ``cdf_state`` to a
            3-length tuple containing 1) the low CDF value for the symbol, 2)
            the high CDF value for the symbol and 3) the updated CDF state.
        inverse_cdf_fun: A ``Callable`` that maps a value and a ``cdf_state``
            to a 4-length tuple containing 1) the low CDF value for the
            symbol, 2) the high value for the symbol, 3) the symbol located at
            the given value, and 4) an updated CDF state.
        precision: An integer specifying codec operating precision.
        message_dtype: Data type for input and output messages.

    Returns:
        A Craystack codec that can be input to the coder.
    """

    def push(symbols, compressed_message, cdf_state):
        cdf_low, cdf_high, cdf_state = cdf_fun(symbols, cdf_state)
        return push_symbols(compressed_message, cdf_low, cdf_high, precision), cdf_state

    def pop(msg_index, compressed_message, message, cdf_state, allow_empty_pops=False):
        cdf_value = peek(compressed_message, precision)
        cdf_low, cdf_high, symbols, cdf_state = inverse_cdf_fun(cdf_value, cdf_state)
        return (
            pop_symbols(
                compressed_message,
                cdf_value,
                cdf_low,
                cdf_high,
                precision,
                allow_empty_pops,
            ),
            lax.dynamic_update_index_in_dim(
                message, symbols.astype(message_dtype), msg_index, 0
            ),
            cdf_state,
        )

    return CrayCodec(push, pop, init_cdf_state, message_dtype, allow_empty_pops)


def bitsback_ans_codec(
    latent_prior_codec: CrayCodec,
    latent_posterior_codec_maker: Callable[[ndarray], CrayCodec],
    obs_codec_maker: Callable[[ndarray], CrayCodec],
    latent_shape: Sequence[int],
    message_dtype: str,
) -> CrayCodec:
    r"""
    Bitsback with Asymmetric Numeral Systems (BB-ANS) craystack codec.

    This function implements BB-ANS as described in 'Practical Lossless
    Compression with Latent Variables using Bits Back Coding', Townsend
    et. al. 2019. Please cite the original paper where appropriate:
    https://arxiv.org/abs/1901.04866

    When compressing a symbol :math:`x` with a latent variable model
    (e.g. VAE), we might not have access to the evidence :math:`p(x)`, only
    latent prior :math:`p(z)`, latent approximate posterior :math:`q(z|x)`,
    and observational contitional likelihood :math:`p(x|z)`. BB-ANS allows
    us to compress with these auxiliary distributions, while achieving a rate
    close to that of compressing with :math:`p(x)`, depending on the quality
    of the approximate posterior :math:`q(z|x)`.

    The encoding scheme for a symbol :math:`x` is
        1. decode :math:`z` with :math:`q(z|x)`
        2. encode :math:`x` with :math:`p(x|z)`
        3. encode :math:`z` with :math:`p(z)`

    Decoding proceeds in reverse order
        3. decode :math:`z` with :math:`p(z)`
        2. decode :math:`x` with :math:`p(x|z)`
        1. encode :math:`z` with :math:`q(z|x)` <-- bits-back step

    Step 1. decreases the ANS state by approximately :math:`\log_2 q(z|x)`,
    while the other 2 steps together increase it by :math:`-\log_2 p(z)p(x|z)`,
    implying the rate is :math:`ELBO(x) = -\log_2 \frac{p(z)p(x|z)}{q(z|x)}`.
    When :math:`q(z|x) = p(z|x)`, i.e. the approximate posterior perfectly
    matches the true posterior, then :math:`ELBO(x) = -\log_2 p(x)`.

    Popping the initial :math`z`'s requires that there be information on the
    ANS stack. This is known as the 'initial bits problem'. The CrayTail
    component of CrayCompressedMessage can automatically return random bits if
    the stack is depleted. For more details, see the ``pop_symbols`` method in
    file ``neuralcompression.entropy_coders.craystack._backend.py``

    Args:
        latent_posterior_codec_maker: A ``Callable`` that receives the symbols
            to be encoded and returns a ``CrayCodec`` representing ``q(z|x)``
            used in step 1.
        obs_codec_maker: A ``Callable``that receives the latents and returns
            a ``CrayCodec` representing ``p(x|z)`` used in step 2.
        latent_prior_codec: A ``CrayCodec`` representing ``p(z)`` in step 3.
        latent_shape: A ``Sequence[int]``, representing the shape of the latents
            ``z``, used to construct a temporary message that holds ``z``
            during encoding and decoding.
        message_dtype: Data type for input and output messages. Muster be the
            same message_dtype as the codecs returned by ``obs_codec_maker``.

    Returns:
        A Craystack codec that can be input to the coder.
    """

    def push(symbols, compressed_message, cdf_state):

        # pop latents with approximate posterior q(latents|symbols)
        latent_posterior_codec = latent_posterior_codec_maker(symbols)
        compressed_message, latents, _ = latent_posterior_codec.pop(
            0,
            compressed_message,
            jnp.zeros(latent_shape, dtype=latent_posterior_codec.message_dtype),
            latent_posterior_codec.cdf_state,
            latent_posterior_codec.allow_empty_pops,
        )

        # push symbols with conditional likelihood p(symbols|latents)
        obs_codec = obs_codec_maker(latents)
        compressed_message, _ = obs_codec.push(
            symbols, compressed_message, obs_codec.cdf_state
        )

        # push latents with prior p(latents)
        compressed_message, _ = latent_prior_codec.push(
            latents,
            compressed_message,
            latent_prior_codec.cdf_state,
        )

        return compressed_message, None

    def pop(msg_index, compressed_message, message, cdf_state):

        # pop latents with prior p(latents)
        compressed_message, latents, _ = latent_prior_codec.pop(
            0,
            compressed_message,
            jnp.zeros(latent_shape, dtype=latent_prior_codec.message_dtype),
            latent_prior_codec.cdf_state,
            latent_prior_codec.allow_empty_pops,
        )

        # pop symbols with conditional likelihood p(symbols|latents)
        obs_codec = obs_codec_maker(latents)
        compressed_message, message, _ = obs_codec.pop(
            msg_index,
            compressed_message,
            message,
            obs_codec.cdf_state,
            obs_codec.allow_empty_pops,
        )
        symbols = message[msg_index]

        # push latents with approximate posterior q(latents|symbols)
        # this is the bits-back step!
        latent_posterior_codec = latent_posterior_codec_maker(symbols)
        compressed_message, _ = latent_posterior_codec.push(
            latents, compressed_message, latent_posterior_codec.cdf_state
        )

        return compressed_message, message, None

    return CrayCodec(push, pop, None, message_dtype)
