# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import jax.numpy as jnp
import numpy as np
import pytest
import scipy

import neuralcompression.entropy_coders


def freqs_to_cdf(freqs, precision=16):
    # Converts a frequency count to a discretized CDF with values
    # between [0, 2**precision)
    pdf = freqs / freqs.sum(axis=-1, keepdims=True)
    cdf = jnp.append(jnp.zeros((*pdf.shape[:-1], 1)), pdf.cumsum(axis=-1), axis=-1)
    return jnp.round(cdf * 2 ** precision).astype(jnp.uint32)


def cdf_to_pdf(cdf):
    # Converts a CDF (discretized, or not) to a PDF
    pdf = jnp.diff(cdf)  # discrete differences
    return pdf / pdf.sum()


def calculate_sample_cdf(alphabet_size, rng, precision=16):
    # Samples frequency counts from a uniform distribution
    # and returns the discretized CDF
    freqs = rng.uniform(size=alphabet_size)
    return freqs_to_cdf(freqs, precision)


def generate_skewed_distribution(alphabet_size, total_counts, seed):
    rng = np.random.default_rng(seed)

    base_counts = []
    for _ in range(alphabet_size):
        base_counts.append(rng.integers(low=1, high=alphabet_size))

    base_counts = np.array(base_counts)
    arr_counts = (base_counts / np.sum(base_counts) * total_counts).astype(np.int32)
    count_diff = total_counts - np.sum(arr_counts)

    if count_diff > 0:
        arr_counts[0] += count_diff
    elif count_diff < 0:
        for ind in range(len(arr_counts)):
            if arr_counts[ind] > -1 * count_diff + 1:
                arr_counts[ind] += count_diff

    return arr_counts


@pytest.mark.parametrize(
    "shape, alphabet_size",
    [((5, 100), 4), ((7, 200), 6), ((1, 100), 20)],
)
def test_arithmetic_coder_identity(shape, alphabet_size):
    # Uses the true source distribution to compress.
    batch_size = shape[0]
    message_len = shape[1]
    seed = 7 * batch_size
    rng = np.random.default_rng(seed)

    messages = jnp.array(
        rng.integers(low=0, high=alphabet_size, size=shape, dtype=np.uint8)
    )
    cdfs = jnp.tile(calculate_sample_cdf(alphabet_size, rng), (batch_size, 1))
    cdf_state = (jnp.array(cdfs),)

    def cdf_fun(symbol, cdf_state):
        return cdf_state[0][symbol], cdf_state[0][symbol + 1], cdf_state

    def inverse_cdf_fun(value, cdf_state):
        symbol = jnp.argmin(value >= cdf_state[0]) - 1
        return cdf_state[0][symbol], cdf_state[0][symbol + 1], symbol, cdf_state

    compressed = neuralcompression.entropy_coders.jac.encode(
        messages, cdf_fun, cdf_state
    )
    decompressed = neuralcompression.entropy_coders.jac.decode(
        compressed, jnp.array([message_len] * batch_size), inverse_cdf_fun, cdf_state
    )

    assert (decompressed == messages).all()


@pytest.mark.parametrize(
    "shape, alphabet_size",
    [((1, 100), 6), ((1, 1500), 3), ((1, 1100), 12)],
)
def test_arithmetic_coder_entropy(shape, alphabet_size):
    # Compresses based on the empirical frequency count of the sequence
    batch_size = shape[0]
    message_len = shape[1]
    seed = 7 * batch_size
    rng = np.random.default_rng(seed)

    messages = jnp.array(
        rng.integers(low=0, high=alphabet_size, size=shape, dtype=np.uint8)
    )

    _, freqs = jnp.unique(messages, return_counts=True)
    cdfs = jnp.expand_dims(freqs_to_cdf(freqs), 0)

    pdf = cdf_to_pdf(cdfs[0])
    entropy = scipy.stats.entropy(pdf, base=2)
    predicted_message_size = int(np.ceil(entropy / 8 * message_len))

    cdf_state = (jnp.array(cdfs),)

    def cdf_fun(symbol, cdf_state):
        return cdf_state[0][symbol], cdf_state[0][symbol + 1], cdf_state

    def inverse_cdf_fun(value, cdf_state):
        symbol = jnp.argmin(value >= cdf_state[0]) - 1
        return cdf_state[0][symbol], cdf_state[0][symbol + 1], symbol, cdf_state

    compressed = neuralcompression.entropy_coders.jac.encode(
        messages, cdf_fun, cdf_state
    )
    decompressed = neuralcompression.entropy_coders.jac.decode(
        compressed, jnp.array([message_len] * batch_size), inverse_cdf_fun, cdf_state
    )

    assert len(compressed[0]) == predicted_message_size
    assert (decompressed == messages).all()


@pytest.mark.parametrize(
    "shape, alphabet_size",
    [((5, 100), 4), ((7, 200), 6), ((1, 100), 20)],
)
def test_arithmetic_coder_adaptive(shape, alphabet_size):
    # Assumes no knowledge of the source distribution. Instead,
    # adaptively estimates the PDF based on the frequency count of
    # previously encoded symbols.
    batch_size = shape[0]
    message_len = shape[1]
    seed = 7 * batch_size
    rng = np.random.default_rng(seed)

    messages = jnp.array(
        rng.integers(low=0, high=alphabet_size, size=shape, dtype=np.uint8)
    )

    cdf_state = (
        jnp.array(rng.integers(low=1, high=42, size=(batch_size, alphabet_size))),
    )

    def cdf_fun(symbol, cdf_state):
        freqs = cdf_state[0]
        cdf = freqs_to_cdf(freqs)
        return cdf[symbol], cdf[symbol + 1], (freqs.at[symbol].add(1),)

    def inverse_cdf_fun(value, cdf_state):
        freqs = cdf_state[0]
        cdf = freqs_to_cdf(freqs)
        symbol = jnp.argmin(value >= cdf) - 1
        return cdf[symbol], cdf[symbol + 1], symbol, (freqs.at[symbol].add(1),)

    compressed = neuralcompression.entropy_coders.jac.encode(
        messages, cdf_fun, cdf_state
    )
    decompressed = neuralcompression.entropy_coders.jac.decode(
        compressed, jnp.array([message_len] * batch_size), inverse_cdf_fun, cdf_state
    )

    assert (decompressed == messages).all()


@pytest.mark.parametrize(
    "base_mass, batch_size, alphabet_size",
    [(1_000, 5, 4), (100, 7, 6), (500, 1, 20)],
)
def test_arithmetic_coder_skewed(base_mass, batch_size, alphabet_size):
    # Tests the AC codec with a highly skewed distribution, i.e, Most of
    # the mass is on the first symbol, e.g. freqs = [999, 1, 1, ...]
    # Uses a typical sequence, i.e. empirical count perfectly matches the pdf.
    # This means the empirical entropy is equal to the true entropy
    freqs = jnp.append(
        base_mass - alphabet_size + 1, jnp.ones(alphabet_size - 1)
    ).astype(int)
    messages = jnp.tile(jnp.repeat(jnp.arange(alphabet_size), freqs), (batch_size, 1))
    cdf_state = (jnp.tile(freqs_to_cdf(freqs), (batch_size, 1)),)

    message_len = messages.shape[1]
    entropy = scipy.stats.entropy(freqs, base=2)
    predicted_message_size = int(np.ceil(entropy / 8 * message_len))

    def cdf_fun(symbol, cdf_state):
        cdf = cdf_state[0]
        return cdf[symbol], cdf[symbol + 1], cdf_state

    def inverse_cdf_fun(value, cdf_state):
        cdf = cdf_state[0]
        symbol = jnp.argmin(value >= cdf) - 1
        return cdf[symbol], cdf[symbol + 1], symbol, cdf_state

    compressed = neuralcompression.entropy_coders.jac.encode(
        messages, cdf_fun, cdf_state
    )
    decompressed = neuralcompression.entropy_coders.jac.decode(
        compressed, jnp.array([message_len] * batch_size), inverse_cdf_fun, cdf_state
    )

    assert (decompressed == messages).all()
    assert len(compressed[0]) == predicted_message_size


@pytest.mark.parametrize(
    "shape, alphabet_size, skewed",
    [
        ((1, 1000, 1), 4, True),
        ((5, 200, 3), 5, False),
        ((5, 600, 20), 200, True),
        ((5, 800, 40), 100, False),
    ],
)
def test_craystack_coder_identity(shape, alphabet_size, skewed):
    # here we only test identity and not message length
    # due to implementation details we don't exactly achieve the
    # entropy-predicted len
    batch_size, message_len, interleave_level = shape
    if skewed:
        freqs = jnp.array(generate_skewed_distribution(alphabet_size, message_len, 124))
        messages = jnp.tile(
            jnp.repeat(jnp.arange(alphabet_size), freqs)[None, :, None],
            (batch_size, 1, interleave_level),
        )
    else:
        rng = np.random.default_rng(123)
        freqs = jnp.ones(alphabet_size) * message_len // alphabet_size
        messages = jnp.array(
            rng.integers(low=0, high=alphabet_size, size=shape, dtype=np.uint8)
        )

    codec = neuralcompression.entropy_coders.craystack.fixed_array_cdf_codec(
        jnp.tile(freqs_to_cdf(freqs), (batch_size, interleave_level, 1)).astype(
            jnp.uint32
        )
    )

    compressed = neuralcompression.entropy_coders.craystack.encode(messages, codec)[0]
    decompressed = neuralcompression.entropy_coders.craystack.decode(
        compressed,
        message_len,
        messages.shape[2:],
        codec,
    )[0]

    assert (decompressed == messages).all()


@pytest.mark.parametrize(
    "shape1, shape2, alphabet_size",
    [
        ((5, 200, 3), (5, 100, 2), 5),
        ((6, 800, 40), (6, 600, 5), 100),
    ],
)
def test_partial_code_decode(shape1, shape2, alphabet_size):
    # here we test encoding messages into a buffer that already has information
    # i.e., multiple sequential encodes
    tail_capacity = 50000
    batch_size, message_len1, interleave_level1 = shape1
    _, message_len2, interleave_level2 = shape2
    rng = np.random.default_rng(123)
    freqs = jnp.ones(alphabet_size) * message_len1 // alphabet_size
    messages1 = jnp.array(
        rng.integers(low=0, high=alphabet_size, size=shape1, dtype=np.uint8)
    )
    messages2 = jnp.array(
        rng.integers(low=0, high=alphabet_size, size=shape2, dtype=np.uint8)
    )

    codec1 = neuralcompression.entropy_coders.craystack.fixed_array_cdf_codec(
        jnp.tile(freqs_to_cdf(freqs), (batch_size, interleave_level1, 1)).astype(
            jnp.uint32
        )
    )

    codec2 = neuralcompression.entropy_coders.craystack.fixed_array_cdf_codec(
        jnp.tile(freqs_to_cdf(freqs), (batch_size, interleave_level2, 1)).astype(
            jnp.uint32
        )
    )

    compressed = neuralcompression.entropy_coders.craystack.encode(
        messages1, codec1, tail_capacity=tail_capacity
    )[0]
    compressed = neuralcompression.entropy_coders.craystack.encode(
        messages2,
        codec2,
        start_buffers=compressed,
        tail_capacity=tail_capacity,
    )[0]
    decompressed2, compressed, _ = neuralcompression.entropy_coders.craystack.decode(
        compressed,
        message_len2,
        messages2.shape[2:],
        codec2,
        tail_capacity=tail_capacity,
    )
    decompressed1, _, _ = neuralcompression.entropy_coders.craystack.decode(
        compressed,
        message_len1,
        messages1.shape[2:],
        codec1,
        tail_capacity=tail_capacity,
    )

    assert (decompressed1 == messages1).all()
    assert (decompressed2 == messages2).all()


@pytest.mark.parametrize(
    "shape, alphabet_size",
    [((5, 100, 1), 4), ((7, 200, 2), 6), ((1, 100, 3), 20)],
)
def test_rans_coder_adaptive(shape, alphabet_size):
    # Assumes no knowledge of the source distribution. Instead,
    # adaptively estimates the PDF based on the frequency count of
    # previously encoded symbols
    batch_size, message_len, interleave_level = shape
    seed = 7 * batch_size
    rng = np.random.default_rng(seed)

    messages = jnp.array(
        rng.integers(low=0, high=alphabet_size, size=shape, dtype=np.int64)
    )

    cdf_state = (
        jnp.array(rng.integers(low=1, high=42, size=(batch_size, alphabet_size))),
    )

    def cdf_fun(symbols, cdf_state):
        freqs = cdf_state[0]
        cdf = freqs_to_cdf(freqs)

        for symbol in symbols:
            freqs.at[symbol].add(1)

        return cdf[symbols], cdf[symbols + 1], (freqs,)

    def inverse_cdf_fun(cdf_value, cdf_state):
        freqs = cdf_state[0]
        cdf = freqs_to_cdf(freqs)
        symbols = jnp.argmin(jnp.expand_dims(cdf_value, -1) >= cdf, axis=-1) - 1

        for symbol in jnp.flip(symbols):
            freqs.at[symbol].add(-1)

        return cdf[symbols], cdf[symbols + 1], symbols, (freqs,)

    codec = neuralcompression.entropy_coders.craystack.default_rans_codec(
        cdf_fun, inverse_cdf_fun, cdf_state
    )

    compressed, cdf_state = neuralcompression.entropy_coders.craystack.encode(
        messages, codec
    )
    codec = neuralcompression.entropy_coders.craystack.default_rans_codec(
        cdf_fun, inverse_cdf_fun, cdf_state
    )
    decompressed = neuralcompression.entropy_coders.craystack.decode(
        compressed,
        message_len,
        messages.shape[2:],
        codec,
    )[0]

    assert (decompressed == messages).all()


@pytest.mark.parametrize(
    "batch_size, message_len, interleave_levels, alphabet_size",
    [(1, 1, 1, 2), (100, 5, 784, 2), (100, 10, 50, 10)],
)
def test_distinct_interleaved_freqs(
    batch_size, message_len, interleave_levels, alphabet_size
):
    # Tests if codecs can handle different frequency counts for
    # each interleaved level.
    rng = np.random.default_rng(123)
    freqs = jnp.array(
        rng.integers(
            low=1,
            high=10,
            size=(batch_size, interleave_levels, alphabet_size),
        )
    )
    codec = neuralcompression.entropy_coders.craystack.fixed_array_cdf_codec(
        freqs_to_cdf(freqs)
    )
    messages = jnp.array(
        rng.integers(
            low=0,
            high=alphabet_size,
            size=(batch_size, message_len, interleave_levels),
            dtype=np.int64,
        )
    )
    decompressed = neuralcompression.entropy_coders.craystack.decode(
        neuralcompression.entropy_coders.craystack.encode(messages, codec)[0],
        message_len,
        messages.shape[2:],
        codec,
    )[0]
    assert (decompressed == messages).all()


@pytest.mark.parametrize(
    (
        "batch_size, message_len, interleave_levels, obs_alphabet_size,"
        "latent_alphabet_size, message_dtype"
    ),
    [(100, 200, 5, 16, 8, "int64"), (100, 50, 20, 4, 20, "uint8")],
)
def test_bitsback_ans_codec_identity(
    batch_size,
    message_len,
    interleave_levels,
    obs_alphabet_size,
    latent_alphabet_size,
    message_dtype,
):
    # Test if compression is perfectly lossless
    # Uses an identical CDF for all interleave levels,
    # but is a function of symbols and latents
    obs_shape = latent_shape = (interleave_levels,)

    rng = np.random.default_rng(123)
    cdf_latent_prior = freqs_to_cdf(  # p(z)
        jnp.array(
            rng.integers(
                low=1,
                high=10,
                size=(*latent_shape, latent_alphabet_size),
            )
        )
    )
    cdf_latent_posterior = freqs_to_cdf(  # q(z|x)
        jnp.array(
            rng.integers(
                low=1,
                high=10,
                size=(obs_alphabet_size, latent_alphabet_size),
            )
        )
    )
    cdf_obs = freqs_to_cdf(  # p(x|z)
        jnp.array(
            rng.integers(
                low=1,
                high=10,
                size=(latent_alphabet_size, obs_alphabet_size),
            )
        )
    )

    messages = jnp.array(
        rng.integers(
            low=0,
            high=obs_alphabet_size,
            size=(batch_size, message_len, *obs_shape),
            dtype=message_dtype,
        )
    )

    latent_prior_codec = (
        neuralcompression.entropy_coders.craystack.fixed_array_cdf_codec(
            cdf_latent_prior, message_dtype=message_dtype
        )
    )

    def latent_posterior_codec_maker(symbols):
        return neuralcompression.entropy_coders.craystack.fixed_array_cdf_codec(
            cdf_latent_posterior[symbols],
            allow_empty_pops=True,
            message_dtype=message_dtype,
        )

    def obs_codec_maker(latents):
        return neuralcompression.entropy_coders.craystack.fixed_array_cdf_codec(
            cdf_obs[latents], message_dtype=message_dtype
        )

    codec = neuralcompression.entropy_coders.craystack.bitsback_ans_codec(
        latent_prior_codec,
        latent_posterior_codec_maker,
        obs_codec_maker,
        latent_shape,
        message_dtype,
    )

    decompressed = neuralcompression.entropy_coders.craystack.decode(
        neuralcompression.entropy_coders.craystack.encode(messages, codec)[0],
        message_len,
        messages.shape[2:],
        codec,
    )[0]
    assert (decompressed == messages).all()
