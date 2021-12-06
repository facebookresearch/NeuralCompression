# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This implementation is based on the blog post at
https://marknelson.us/posts/2014/10/19/data-compression-with-arithmetic-coding.html
"""

from typing import Any, Callable, List, Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.numpy import ndarray

jax.config.update("jax_enable_x64", True)

MAX_CODE = 0xFFFFFFFF
ONE_HALF = 0x80000000
ONE_FOURTH = 0x40000000
THREE_FOURTHS = 0xC0000000
PREC_COUNT = 0x10000


def _append_bit(
    bit: bool, int_array: ndarray, byte: int, idx: int, bit_idx: int
) -> Tuple[ndarray, int, int, int]:
    """Append a single bit to the integer bit array."""

    def true_branch(vals):
        return jax.ops.index_update(vals[0], vals[1], vals[2]), jnp.uint8(0)

    def false_branch(vals):
        return vals[0], vals[2]

    byte <<= 1
    byte |= bit

    bit_idx += 1
    int_array, byte = lax.cond(
        bit_idx == 8, true_branch, false_branch, (int_array, idx, byte)
    )
    idx += bit_idx // 8
    bit_idx = jnp.mod(bit_idx, 8)

    return int_array, byte, idx, bit_idx


def _append_bit_and_pending(
    bit: bool, pending_bits: int, int_array: ndarray, byte: int, idx: int, bit_idx: int
) -> Tuple[ndarray, int, int, int]:
    """Append a single bit and a pending_bits rev. bits to an integer array."""

    def add_pending_bit(vals):
        pending_bits, bit, int_array, byte, idx, bit_idx = vals

        int_array, byte, idx, bit_idx = _append_bit(
            jnp.logical_not(bit), int_array, byte, idx, bit_idx
        )
        pending_bits -= 1

        return pending_bits, bit, int_array, byte, idx, bit_idx

    def condition(vals):
        return vals[0] > 0

    int_array, byte, idx, bit_idx = _append_bit(bit, int_array, byte, idx, bit_idx)

    _, _, int_array, byte, idx, bit_idx = lax.while_loop(
        condition, add_pending_bit, (pending_bits, bit, int_array, byte, idx, bit_idx)
    )

    return int_array, byte, idx, bit_idx


def _retrieve_bit(idx: int, bit_idx: int, int_array: ndarray) -> Tuple[int, int, int]:
    """Retrieve a single bit from an integer array."""
    cur_int = int_array[idx]
    bit = ((cur_int >> (7 - bit_idx)) & 1) > 0
    idx += (bit_idx + 1) // 8
    bit_idx = jnp.mod(bit_idx + 1, 8)

    return bit, idx, bit_idx


def _enqueue(
    symbol: Any,
    int_array: ndarray,
    byte: int,
    idx: ndarray,
    bit_idx: ndarray,
    pending_bits: ndarray,
    high: ndarray,
    low: ndarray,
    cdf_fun: Callable,
    cdf_state: Any,
    precision: ndarray,
) -> Tuple[ndarray, int, ndarray, ndarray, ndarray, ndarray, ndarray, Any]:
    """
    Enqueue a single symbol into the compressed int_array.

    Args:
        symbol: A single symbol to code into ``int_array``.
        int_array: A 8-bit integer array holding the so-far compressed message.
        byte: The current operating byte of ``int_array``.
        idx: The index of `byte`'s location within ``int_array``.
        bit_idx: The index of the current bit within ``byte``.
        pending_bits: The number of pending bits to encode (arises from
            convergent CDF).
        high: The current high of the range.
        low: The current low of the range.
        cdf_fun: A callable that can return the probability of the symbol given
            a ``cdf_state``.
        cdf_state: The current state of the CDF function (contains CDF array or
            can be used for conditional probabilites).
        precision: An integer specifying the operating precision of the
            algorithm and max CDF value.

    Returns:
        A tuple containing updated versions of ``int_array``, ``byte``,
            ``idx``, ``bit_idx``, ``pending_bits``, ``high``, ``low``, and
            ``cdf_state``.
    """
    # jax functions - these will all be jit-compiled for our while loop
    def decrease_high(vals):
        """Based on high decreasing, we append a 0 most significant bit."""
        high, low, pending_bits, int_array, byte, idx, bit_idx = vals
        int_array, byte, idx, bit_idx = _append_bit_and_pending(
            False, pending_bits, int_array, byte, idx, bit_idx
        )
        pending_bits *= 0

        return high, low, pending_bits, int_array, byte, idx, bit_idx

    def increase_low(vals):
        """Based on low increasing, we append a 1 most significant bit."""
        high, low, pending_bits, int_array, byte, idx, bit_idx = vals
        int_array, byte, idx, bit_idx = _append_bit_and_pending(
            True, pending_bits, int_array, byte, idx, bit_idx
        )
        pending_bits *= 0

        return high, low, pending_bits, int_array, byte, idx, bit_idx

    def renormalize_range(vals):
        """Range is too low for numerical precision - this fn readjusts."""
        high, low, pending_bits, _, _, _, _ = vals
        pending_bits += 1
        low -= ONE_FOURTH
        high -= ONE_FOURTH

        return (high, low, pending_bits) + vals[3:]

    branches = [decrease_high, increase_low, renormalize_range]

    def choose_branch_and_execute(vals):
        """if/else bad for jax, so we use a switch with argmin instead."""
        high, low, pending_bits, int_array, byte, idx, bit_idx = vals

        # this will return true for the first one of these that is true
        branch_idx = lax.argmax(
            jnp.array(
                [
                    (high < ONE_HALF),
                    (low >= ONE_HALF),
                    (low >= ONE_FOURTH) * (high < THREE_FOURTHS),
                ]
            ),
            0,
            jnp.int32,
        )

        # execute chosen branch
        high, low, pending_bits, int_array, byte, idx, bit_idx = lax.switch(
            branch_idx,
            branches,
            (high, low, pending_bits, int_array, byte, idx, bit_idx),
        )

        # perform after-shifts
        high <<= 1
        high += 1
        low <<= 1
        high &= MAX_CODE
        low &= MAX_CODE

        return high, low, pending_bits, int_array, byte, idx, bit_idx

    def condition(vals):
        """Once our range has converged, we break the while loop."""
        return (
            (vals[0] < ONE_HALF)
            + (vals[1] >= ONE_HALF)
            + (vals[1] >= ONE_FOURTH) * (vals[0] < THREE_FOURTHS)
        ) > 0

    # calculate our current range window
    span = jnp.uint64(high) - jnp.uint64(low) + 1

    # get the low and high points of the CDF for current symbol
    range_low, range_high, cdf_state = cdf_fun(symbol, cdf_state)

    # narrow the range using cdf values
    high = (
        low + jnp.uint32((span * jnp.uint64(range_high)) >> jnp.uint64(precision)) - 1
    )
    low = low + jnp.uint32((span * jnp.uint64(range_low)) >> jnp.uint64(precision))

    # build and execute while loop
    high, low, pending_bits, int_array, byte, idx, bit_idx = lax.while_loop(
        condition,
        choose_branch_and_execute,
        (high, low, pending_bits, int_array, byte, idx, bit_idx),
    )

    return int_array, byte, idx, bit_idx, pending_bits, high, low, cdf_state


def _dequeue(
    value: ndarray,
    int_array: ndarray,
    idx: ndarray,
    bit_idx: ndarray,
    high: ndarray,
    low: ndarray,
    inverse_cdf_fun: Callable,
    inverse_cdf_state: Any,
    precision: ndarray,
) -> Tuple[ndarray, Any, ndarray, ndarray, ndarray, ndarray, Any]:
    """
    Dequeue a single symbol from the compressed ``int_array``.

    Args:
        value: The current location within the CDF range used to decode symbol.
        int_array: A 8-bit integer array holding the so-far compressed message.
        idx: The index of the operating location within `int_array`.
        bit_idx: The index of the current bit within the operating byte.
        high: The current high of the range.
        low: The current low of the range.
        inverse_cdf_fun: A callable that can return a symbol, its low and high
            probabilities, and an updated `inverse_cdf_state` given a value and
            the current ``inverse_cdf_state``.
        inverse_cdf_state: The current state of the CDF function (contains CDF
            array or can be used for conditional probabilites).
        precision: An integer specifying the operating precision of the
            algorithm and max CDF value.

    Returns:
        A tuple containing updated versions of ``value``, ``high``, ``low``,
            ``idx``, ``bit_idx``, ``int_array``.
    """
    # jax functions - these will all be jit-compiled for our while loop
    def high_cross_half(vals):
        """No need to do anything, bit is 0."""
        return vals

    def low_cross_half(vals):
        """Based on low increasing, adjust limits."""
        value, high, low = vals

        value -= ONE_HALF
        low -= ONE_HALF
        high -= ONE_HALF

        return value, high, low

    def cross_fourth(vals):
        """Range got too narrow, open up."""
        value, high, low = vals

        value -= ONE_FOURTH
        low -= ONE_FOURTH
        high -= ONE_FOURTH

        return value, high, low

    branches = [high_cross_half, low_cross_half, cross_fourth]

    def choose_branch_and_execute(vals):
        """if/else bad for jax, so we use a switch with argmin instead."""
        value, high, low, idx, bit_idx, int_array = vals

        # this will return true for the first one of these that is true
        branch_idx = lax.argmax(
            jnp.array(
                [
                    (high < ONE_HALF),
                    (low >= ONE_HALF),
                    (low >= ONE_FOURTH) * (high < THREE_FOURTHS),
                ]
            ),
            0,
            jnp.int32,
        )

        # execute chosen branch
        value, high, low = lax.switch(branch_idx, branches, (value, high, low))

        low <<= 1
        high <<= 1
        high += 1
        value <<= 1
        bit, idx, bit_idx = _retrieve_bit(idx, bit_idx, int_array)
        value += bit * (idx < len(int_array))

        return value, high, low, idx, bit_idx, int_array

    def condition(vals):
        """Once our range has converged, we break the while loop."""
        return (
            (vals[1] < ONE_HALF)
            + (vals[2] >= ONE_HALF)
            + (vals[2] >= ONE_FOURTH) * (vals[1] < THREE_FOURTHS)
        ) > 0

    # calculate our current range window
    span = jnp.uint64(high) - jnp.uint64(low) + 1

    scaled_value = jnp.uint16(
        ((jnp.uint64(value) - jnp.uint64(low) + 1) * jnp.uint64(PREC_COUNT) - 1) / span
    )

    # get the low and high points of the CDF for current symbol
    range_low, range_high, symbol, inverse_cdf_state = inverse_cdf_fun(
        scaled_value, inverse_cdf_state
    )

    # narrow the range using cdf values
    high = (
        low + jnp.uint32((span * jnp.uint64(range_high)) >> jnp.uint64(precision)) - 1
    )
    low = low + jnp.uint32((span * jnp.uint64(range_low)) >> jnp.uint64(precision))

    # build and execute while loop
    value, high, low, idx, bit_idx, _ = lax.while_loop(
        condition,
        choose_branch_and_execute,
        (value, high, low, idx, bit_idx, int_array),
    )

    return value, symbol, idx, bit_idx, high, low, inverse_cdf_state


def _jax_encode(
    message: jnp.ndarray, cdf_fun: Callable, init_cdf_state: Any, buffer_oversize: float
) -> Tuple[ndarray, int]:
    """JAX encoding backend function."""
    # array initialization
    idx = jnp.array(0, jnp.int32)
    bit_idx = jnp.array(0, jnp.int32)
    int_array = jnp.zeros(
        int(buffer_oversize * len(message)), dtype=jnp.uint8
    )  # message can grow
    high = jnp.array(MAX_CODE, dtype=jnp.uint32)
    low = jnp.array(0, dtype=jnp.uint32)
    precision = jnp.array(16, dtype=jnp.uint32)
    pending_bits = jnp.array(0, dtype=jnp.int32)

    # loop iterate - this enqueues one symbol into the compressed array
    def enqueue_one_symbol(msg_index, enqueue_state):
        symbol = message[msg_index]
        (
            int_array,
            byte,
            idx,
            bit_idx,
            pending_bits,
            high,
            low,
            cdf_state,
        ) = enqueue_state
        return _enqueue(
            symbol,
            int_array,
            byte,
            idx,
            bit_idx,
            pending_bits,
            high,
            low,
            cdf_fun,
            cdf_state,
            precision,
        )

    # loop over every symbol in the message and add it to the compressed array
    int_array, byte, idx, bit_idx, pending_bits, _, low, _ = lax.fori_loop(
        0,
        len(message),
        enqueue_one_symbol,
        (
            int_array,
            int_array[0],
            idx,
            bit_idx,
            pending_bits,
            high,
            low,
            init_cdf_state,
        ),
    )

    # append the final trailing bits
    pending_bits += 1

    def append_final_0(final_append):
        pending_bits, int_array, byte, idx, bit_idx = final_append
        int_array, byte, idx, bit_idx = _append_bit_and_pending(
            False, pending_bits, int_array, byte, idx, bit_idx
        )
        return int_array, byte, idx, bit_idx

    def append_final_1(final_append):
        pending_bits, int_array, byte, idx, bit_idx = final_append
        int_array, byte, idx, bit_idx = _append_bit_and_pending(
            True, pending_bits, int_array, byte, idx, bit_idx
        )
        return int_array, byte, idx, bit_idx

    int_array, byte, idx, bit_idx = lax.cond(
        low < ONE_FOURTH,
        append_final_0,
        append_final_1,
        (pending_bits, int_array, byte, idx, bit_idx),
    )

    int_array = jax.ops.index_update(int_array, idx, byte << (8 - bit_idx))

    return int_array, idx


def _jax_decode(
    message_len: ndarray,
    int_array: ndarray,
    inverse_cdf_fun: Callable,
    init_inverse_cdf_state: Any,
) -> ndarray:
    """JAX decoding backend function."""
    idx = jnp.array(0, jnp.int32)
    bit_idx = jnp.array(0, jnp.int32)
    message = jnp.zeros(message_len, dtype=jnp.uint8)
    value = jnp.array(0, jnp.uint32)
    high = jnp.array(MAX_CODE, dtype=jnp.uint32)
    low = jnp.array(0, dtype=jnp.uint32)
    precision = jnp.array(16, dtype=jnp.uint32)

    def dequeue_one_symbol(msg_idx, dequeue_state):
        message = dequeue_state[0]
        value, symbol, idx, bit_idx, high, low, inverse_cdf_state = _dequeue(
            *dequeue_state[1:7], inverse_cdf_fun, *dequeue_state[7:]
        )

        message = jax.ops.index_update(message, msg_idx, symbol)

        return (
            message,
            value,
            int_array,
            idx,
            bit_idx,
            high,
            low,
            inverse_cdf_state,
            precision,
        )

    # read out one bit and append it to our byte
    def append_bit_to_value(loop_idx, vals):
        value, idx, bit_idx, int_array = vals

        bit, idx, bit_idx = _retrieve_bit(idx, bit_idx, int_array)

        value <<= 1
        value += bit

        return value, idx, bit_idx, int_array

    value, idx, bit_idx, _ = lax.fori_loop(
        0,
        jnp.int64(precision * 2),
        append_bit_to_value,
        (value, idx, bit_idx, int_array),
    )

    # extract all symbols from the compressed array
    message = lax.fori_loop(
        0,
        message_len,
        dequeue_one_symbol,
        (
            message,
            value,
            int_array,
            idx,
            bit_idx,
            high,
            low,
            init_inverse_cdf_state,
            precision,
        ),
    )[0]

    return message


def _single_batch_encode(
    message: jnp.ndarray, cdf_fun: Callable, init_cdf_state: Any, buffer_oversize: float
) -> Tuple[ndarray, int]:
    """Encodes one index of a batch."""
    bits, idx = _jax_encode(message, cdf_fun, init_cdf_state, buffer_oversize)

    return bits, idx


def _single_batch_decode(
    data: ndarray,
    message_len: ndarray,
    inverse_cdf_fun: Callable,
    init_inverse_cdf_state: Any,
) -> jnp.ndarray:
    """Decodes one index of a batch."""
    message = _jax_decode(
        message_len=message_len,
        int_array=data,
        inverse_cdf_fun=inverse_cdf_fun,
        init_inverse_cdf_state=init_inverse_cdf_state,
    )

    return message


def encode(
    messages: ndarray,
    cdf_fun: Callable,
    init_cdf_states: Any,
    buffer_oversize: float = 1.5,
) -> List[bytes]:
    """
    Encode a batch of messages via sequential arithmetic coding.

    This function encodes a batch of messages into a list of byte arrays using
    cumulative distribution Python functionals. Its unique advantage is that
    the CDF can be specified functionally rather than via look-up table, which
    allows use of complicated functionals such as neural networks.

    * ``messages`` should be a JAX-Numpy array of shape
      ``(batch_size, message_length)``. The computation will be parallelized
      over batches.
    * ``cdf_fun`` should be a callable of the form
      ``f(symbol, cdf_state) -> (cdf_low, cdf_high, cdf_state)``, where
      ``cdf_low`` is :math:`P(x <= symbol)` and `cdf_high` is
      :math:`P(x <= symbol+1)`. ``cdf_state`` can be any collection of JAX
      arrays that is used internally by ``cdf_fun`` during coding. The CDF
      probabilities should be on an integer scale between ``0`` and
      ``2 ** precision``, where precision is 16 (for a 16-bit working integer).

    Args:
        messages: An input JAX ndarray where the first dimension is the batch
            dimension.
        cdf_fun: A Python callable for calculating symbol probabilities.
        init_cdf_state: The initialization state of the CDF function (contains
            CDF array or can be used for conditional probabilites).
        buffer_oversize: The compression buffer size is
            `buffer_oversize*len(messages[0])`.

    Returns:
        A list of length ``batch_size`` where each element in the list is a
            compressed byte array of the content in that batch index.
    """

    def run_encode(message, init_cdf_state):
        return _single_batch_encode(message, cdf_fun, init_cdf_state, buffer_oversize)

    map_fun = jax.vmap(run_encode)

    bits, num_bytes = map_fun(messages, init_cdf_states)

    if any([i > len(b) for b, i in zip(bits, num_bytes)]):
        raise RuntimeError("Message buffer overrun - try increasing buffer_oversize.")

    bits = [bytes(b[: i + 1]) for b, i in zip(bits, num_bytes)]

    return bits


def decode(
    data_sets: List[bytes],
    message_lens: ndarray,
    inverse_cdf_fun: Callable,
    init_inverse_cdf_states: Any,
) -> ndarray:
    """
    Decode a batch of messages via sequential arithmetic coding.

    This function decodes a list of byte arrays to a batch of messages into a
    using cumulative distribution Python functionals. Its unique advantage is
    that the CDF can be specified functionally rather than via look-up table,
    which allows use of complicated functionals such as neural networks.

    * ``data_sets`` should be a list of Python byte arrays representing
      compressed data.
    * ``inverse_cdf_fun`` should be a callable of the form
      ``f(value, inverse_cdf_state) -> (cdf_low, cdf_high, symbol,
      inverse_cdf_state)``, where `cdf_low` is :math:`P(x <= symbol)` and
      ``cdf_high`` is :math:`P(x <= symbol+1)`. `inverse_cdf_state` can be any
      collection of JAX arrays that is used internally by `inverse_cdf_fun`
      during coding. The CDF probabilities should be on an integer scale
      between ``0` and ``2 ** precision``, where precision is 16 (for a 16-bit
      working integer).

    Args:
        data_sets: A list of byte arrays containing compressed messages.
        message_lens: An array of lengths for the decoded messages.
        inverse_cdf_fun: A Python callable that calculates the symbol,
            low, and high CDF cut points based on an input probability.
        init_inverse_cdf_state: The initialization for the state of the inverse
            CDF function (contains CDF array or can be used for conditional
            probabilites).

    Returns:
        A batch of decoded messages. If the messages all have the same length,
            an ndarray is returned. Otherwise, the function will return a list
            of ndarray variables.
    """
    max_message_len = jnp.max(message_lens)

    def run_decode(data, init_inverse_cdf_state):
        return _single_batch_decode(
            data, max_message_len, inverse_cdf_fun, init_inverse_cdf_state
        )

    map_fun = jax.vmap(run_decode)

    # make all byte arrays the same size so we can group them in one jax array
    max_len = max(len(d) for d in data_sets)
    data_sets = jnp.stack(
        [
            jnp.append(
                jnp.array(bytearray(b), dtype=jnp.uint8),
                jnp.zeros(max_len - len(b), dtype=jnp.uint8),
            )
            for b in data_sets
        ]
    )

    # dispatch
    messages = map_fun(data_sets, init_inverse_cdf_states)

    if any([(not ml == max_message_len) for ml in message_lens]):
        messages = [m[:ml] for m, ml in zip(messages, message_lens)]

    return messages
