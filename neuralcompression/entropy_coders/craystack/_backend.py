# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2021 Jamie Townsend
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Code modified from
https://github.com/j-towns/crayjax
"""

from typing import NamedTuple, Sequence, Tuple

import jax.lax as lax
import jax.numpy as jnp
from jax.numpy import ndarray
from jax.random import PRNGKey, randint

HEAD_PRECISION, HEAD_DTYPE = 32, "uint32"
TAIL_PRECISION, TAIL_DTYPE = 8, "uint8"
HEAD_MIN = 1 << HEAD_PRECISION - TAIL_PRECISION
TAIL_DEFAULT_PRNGKEY = PRNGKey(1337)
TAIL_MAX = (1 << TAIL_PRECISION) - 1


class CrayTail(NamedTuple):
    """
    Type for hahndling craystack API compressed message tail.

    The Craystack API decomposes a compressed message into a head and a tail.
    In JAX the tail is implemented as a 2-tuple with an int that points to the
    current index in the tail.

    The Craystack message structure is described in `A tutorial on the range
    variant of asymmetric numeral systems (J. Townsend)
    <https://arxiv.org/abs/2001.09186>`_.

    Args:
        pointer: An integer pointing to the current end of data written
            to the end of the array.
        data: An array of tail data.
    """

    pointer: ndarray
    data: ndarray


class CrayCompressedMessage(NamedTuple):
    """
    Type for handling cray-stack API compressed messages.

    The Craystack API decomposes a compressed message into a head and a tail.
    In JAX the tail is implemented as a 2-tuple with an int that points to the
    current index in the tail.

    The Craystack message structure is described in `A tutorial on the range
    variant of asymmetric numeral systems (J. Townsend)
    <https://arxiv.org/abs/2001.09186>`_.

    Args:
        head: Craystack head, typically an array of unsigned integers
            (typically 32-bit) taking the shape of the message.
        tail: Craystack tail, a tuple with a variable-length array of 8-bit
            unsigned integers and an integer that points to the current size
            of data written to the array.
    """

    head: ndarray
    tail: CrayTail


def empty_message(shape: Sequence[int], tail_capacity: int) -> CrayCompressedMessage:
    """
    Creates an empty message with the head initialized to the min value.

    Args:
        shape: Shape of message to create, excluding batch dimensions and
            message length.
        tail_capacity: Size of the Craystack tail.

    Returns:
        An empty base message.
    """
    return CrayCompressedMessage(
        jnp.full(shape, HEAD_MIN, HEAD_DTYPE),
        empty_stack(tail_capacity),
    )


def empty_stack(capacity: int, prng_key: ndarray = TAIL_DEFAULT_PRNGKEY) -> CrayTail:
    """Returns a tuple with the size of the stack and a stack array."""
    return CrayTail(
        jnp.array([capacity]), randint(prng_key, (capacity,), 0, TAIL_MAX, TAIL_DTYPE)
    )


def tail_push(tail: CrayTail, push_indices: ndarray, data: ndarray) -> CrayTail:
    """
    Push data to tail at specified indices.

    Args:
        tail: A CrayTail to push data to.
        push_indices: Indices specifying push locations (for interleaved
            coding).

    Returns:
        ``tail`` with data inserted at ``push_indices`` and tail pointer moved
            to end of new data.
    """
    push_indices, data = jnp.ravel(push_indices), jnp.ravel(data)
    limit, tail_data = tail
    # insert current information batch into the tail
    return CrayTail(
        limit - push_indices.sum(),
        lax.dynamic_update_slice(
            tail_data, lax.sort_key_val(push_indices, data)[1], limit - data.size
        ),
    )


def tail_pop(
    tail: CrayTail, pop_indices: ndarray, allow_empty_pop: bool = False
) -> Tuple[CrayTail, ndarray]:
    """
    Pop data from tail at specified indices.

    Args:
        tail: A CrayTail to pop data from.
        pop_indices: Indices specifying pop locations (for interleaved
            coding).
        allows_empty_pops: A boolean that specifys if random bits should
            be created on demand, if the stack is depleted.

    Returns:
        A 2-tuple containing:
            ``tail`` with data popped from top of stack by moving the tail
                pointer.
            The information content that was popped.

    """
    pop_indices_flat = jnp.ravel(pop_indices)  # interleave flattening
    limit, data = tail
    unsorted = lax.sort_key_val(pop_indices_flat, jnp.arange(pop_indices.size))[1]

    limit = lax.cond(
        allow_empty_pop,
        lambda _: jnp.clip(limit + pop_indices.sum(), 0, len(data)),
        lambda _: limit + pop_indices.sum(),
        None,
    )
    # here we return a slice containing the information in the pop indices
    # we don't actually need to pop the values from the array - instead we just
    # move the pointer along and remove the popped values at the end
    return CrayTail(limit, data), jnp.reshape(
        lax.sort_key_val(
            unsorted,
            lax.dynamic_slice(
                data, limit - pop_indices.size, pop_indices_flat.shape  # type: ignore
            ),
        )[1],
        pop_indices.shape,
    )


def push_symbols(
    compressed_message: CrayCompressedMessage,
    cdf_low: ndarray,
    cdf_high: ndarray,
    precision: int,
) -> CrayCompressedMessage:
    """
    Pushes information corresponding to a symbol on the top of the stack.

    Args:
        compressed_message: Input stack to have data pushed to.
        cdf_low: The low CDF value, e.g., ``cdf[symbol]``.
        cdf_high: The high CDF, e.g., ``cdf[symbol+1]``.
        precision: Operating precision for push operations.

    Returns:
        Compressed message with information content pushed to top of stack.
    """
    head, tail = compressed_message
    frequencies = cdf_high - cdf_low

    # Why three writes you may ask?
    # The answer is to vectorize. In the pop function normally this is done
    # with a while loop that would renormalize the range in steps of 8 bits.
    # Since we're using 32 bits for the head the increments can come at levels
    # of 8, 16, or 24, depending on the starting value of the head. Since the
    # push operation is the inverse, we have three possible pushes.

    indices = head >> 2 * TAIL_PRECISION + HEAD_PRECISION - precision >= frequencies
    tail = tail_push(tail, indices, head.astype(TAIL_DTYPE))
    head = jnp.where(indices, head >> TAIL_PRECISION, head)

    indices = head >> 1 * TAIL_PRECISION + HEAD_PRECISION - precision >= frequencies
    tail = tail_push(tail, indices, head.astype(TAIL_DTYPE))
    head = jnp.where(indices, head >> TAIL_PRECISION, head)

    indices = head >> HEAD_PRECISION - precision >= frequencies
    tail = tail_push(tail, indices, head.astype(TAIL_DTYPE))
    head = jnp.where(indices, head >> TAIL_PRECISION, head)

    head_div_freqs, head_mod_freqs = jnp.divmod(head, frequencies)

    return CrayCompressedMessage(
        (head_div_freqs << precision) + head_mod_freqs + cdf_low, tail
    )


def pop_symbols(
    compressed_message: CrayCompressedMessage,
    cfs: int,
    cdf_low: int,
    cdf_high: int,
    precision: int,
    allow_empty_pops: bool = False,
) -> CrayCompressedMessage:
    """
    Pops information corresponding to single symbol from the top of the stack.

    Args:
        compressed_message: Input stack to have data popped from.
        cfs: The CDF symbol value.
        cdf_low: The low CDF value, e.g., ``cdf[symbol]``.
        cdf_high: The high CDF, e.g., ``cdf[symbol+1]``.
        precision: Operating precision for pop operations.

    Returns:
        Compressed message with information content popped from top of stack.
    """
    frequencies = cdf_high - cdf_low
    head = jnp.array(
        frequencies * (compressed_message[0] >> precision) + cfs - cdf_low,
        dtype=HEAD_DTYPE,
    )
    tail = compressed_message[1]

    # Why range over three you may ask?
    # The answer is to vectorize. In the pop function normally this is done
    # with a while loop that would renormalize the range in steps of 8 bits.
    # Since we're using 32 bits for the head the increments can come at levels
    # of 8, 16, or 24, depending on the starting value of the head.

    for _ in range(3):
        indices = jnp.less(head, HEAD_MIN)
        tail, new_head = tail_pop(tail, indices, allow_empty_pops)
        head = jnp.where(indices, head << TAIL_PRECISION | new_head, head)

    return CrayCompressedMessage(head.astype(HEAD_DTYPE), tail)


def peek(compressed_message: CrayCompressedMessage, precision: int) -> int:
    """
    Look at the top of the stack without popping.

    Args:
        compressed_message: Stack to peek into top of.
        precision: Precision with which to unpack Craystack head.

    Returns:
        Current CDF value from top of stack.
    """
    head = compressed_message[0]
    return head & ((1 << precision) - 1)


def craymessage_to_array(
    compressed_message: CrayCompressedMessage,
) -> Tuple[ndarray, int]:
    """
    Flatten a CrayCompressedMessage to a single array.

    Args:
        compressed_message: Message to be flattened.

    Returns:
        A 2-tuple with:
            A flattened version of compressed_message.
            An integer specifying the message length.
    """
    head, ([tail_limit], tail_data) = compressed_message
    head = jnp.ravel(head)
    return (
        jnp.concatenate(
            [
                (head >> 3 * TAIL_PRECISION).astype(TAIL_DTYPE),  # type: ignore
                (head >> 2 * TAIL_PRECISION).astype(TAIL_DTYPE),  # type: ignore
                (head >> TAIL_PRECISION).astype(TAIL_DTYPE),  # type: ignore
                head.astype(TAIL_DTYPE),
                tail_data,
            ]
        ),
        head.shape[0] * 4 + tail_limit,
    )


def array_to_craymessage(
    compressed_message: ndarray, shape: Sequence[int], tail_limit: int
) -> CrayCompressedMessage:
    """
    Convert a flattened message back to CrayCompressedMessage format.

    Args:
        compressed_message: Message to be unflattened.
        shape: Shape of original message (specifies shape of the head).
        tail_limit: Size of the tail.

    Returns:
        Message in standard CrayCompressedMessage format.
    """
    size = int(jnp.prod(jnp.array(shape)))
    head_highest, head_high, head_low, head_lowest, tail = jnp.split(
        compressed_message, [size, 2 * size, 3 * size, 4 * size]
    )
    head = (
        head_highest.astype(HEAD_DTYPE) << 3 * TAIL_PRECISION
        | head_high.astype(HEAD_DTYPE) << 2 * TAIL_PRECISION
        | head_low.astype(HEAD_DTYPE) << 1 * TAIL_PRECISION
        | head_lowest.astype(HEAD_DTYPE)
    )
    tail = CrayTail(jnp.array([tail_limit]), tail)

    return CrayCompressedMessage(jnp.reshape(head, size), tail)


def insert_zeros(
    message: ndarray, shape: Sequence[int], tail_capacity: int
) -> Tuple[ndarray, int]:
    """Insert zeros between head and tail based on tail capacity."""
    head_size = int(jnp.prod(jnp.array(shape)) * 4)
    splits = jnp.split(message, [head_size])
    needed_zeros = tail_capacity - splits[1].shape[0]
    tail_limit = needed_zeros + 1

    if needed_zeros < 0:
        raise ValueError("tail_capacity insufficient for needed buffer size.")

    return (
        jnp.concatenate((splits[0], jnp.zeros(needed_zeros, TAIL_DTYPE), splits[1])),
        tail_limit,
    )


def convert_to_embedded(
    compressed_messages: Sequence[ndarray],
    message_shape: Sequence[int],
    tail_capacity: int,
) -> Tuple[ndarray, ndarray]:
    """
    Embed a truncated list of byte arrays into equal-size arrays.

    Args:
        compressed_messages: Packed message to embed into a standard-length.
        message_shape: Shape of unpacked message.
        tail_capacity: Desired size of the tail.

    Returns:
        A 2-tuple containing:
            A full-length stack with the message embedded into constant sizes
                via zero-padding.
            A list of integers pointing to the end of each tail.
    """
    embedded_messages = []
    tail_limits = []
    for cm in compressed_messages:
        cm_conv, tail_limit = insert_zeros(cm, message_shape, tail_capacity)
        embedded_messages.append(cm_conv)
        tail_limits.append(tail_limit)

    return jnp.stack(embedded_messages), jnp.stack(tail_limits)
