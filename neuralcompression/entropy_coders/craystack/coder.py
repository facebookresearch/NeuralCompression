# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Sequence, Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.numpy import ndarray

from ._backend import (
    CrayCompressedMessage,
    array_to_craymessage,
    convert_to_embedded,
    craymessage_to_array,
    empty_message,
)
from .codecs import CrayCodec


def _jax_encode(
    message: ndarray,
    compressed_message: CrayCompressedMessage,
    codec: CrayCodec,
    cdf_state: Sequence[ndarray],
) -> Tuple[Tuple[ndarray, int], Sequence[ndarray]]:
    """
    JAX rANS encoding function.

    At a high level, this function takes a message and appends the information
    content of that message to a byte stack array. The size and content of the
    append operation depends on the probability of each symbol in the message
    occurring.

    Args:
        message: The array to be encoded.
        compressed_message: A stack byte array to append the compressed symbols
            to.
        codec: A named tuple object containing functions for push and pop
            operations, as well as an initial state fo the CDF functions (for
            context-adaptive coding) and a data type specification for the
            message.
        cdf_state: The initialization state of the CDF function
            (contains CDF array or can be used for conditional probabilites).

    Returns:
        A 2-tuple containing:
            A stack with the information content of the compressed symbols.
            The final CDF state.
    """
    message_len = message.shape[0]

    def push_one_symbol(msg_index, vals):
        return codec.push(
            message[message_len - msg_index - 1], *vals
        )  # rANS encodes backwards

    compressed_message_final, cdf_state_final = lax.fori_loop(
        0, message_len, push_one_symbol, (compressed_message, cdf_state)
    )

    return craymessage_to_array(compressed_message_final), cdf_state_final


def _jax_decode(
    compressed_message: ndarray,
    tail_limit: int,
    message_len: int,
    message_shape: Sequence[int],
    codec: CrayCodec,
    cdf_state: Sequence[ndarray],
) -> Tuple[Tuple[ndarray, int], ndarray, Sequence[ndarray]]:
    """
    JAX rANS decoding function.

    At a high level, this function takes a stack of information
    (``compressed_message``) and peeks at the top of the stack to see what the
    current symbol is. After identifying the symbol, this function pops a
    number of bits from the top of the stack approximately equal to the
    information content of the symbol (i.e. ``-log(symbol probability)``). This
    is done ``message_len`` times until the full message is retrieved.

    Args:
        compressed_message: The input stack containing the compressed meessage.
        tail_limit: A pointer to the current end of the tail.
        message_len: The size of the message to be decoded.
        message_shape: The message shape containing the interleaved dimension
            size.
        codec: A named tuple object containing functions for push and pop
            operations, as well as an initial state fo the CDF functions (for
            context-adaptive coding) and a data type specification for the
            message.
        cdf_state: The initialization state of the inverse CDF function
            (contains CDF array or can be used for conditional probabilites).

    Returns:
        A 3-tuple containing:
            The decoded messages of size
                    ``(message_len, *message_shape)``.
            A byte array of compressed data after removing the target
                message.
            The final CDF state.
    """
    message = jnp.zeros((message_len, *message_shape), dtype=codec.message_dtype)

    def pop_one_symbol(msg_index, vals):
        return codec.pop(msg_index, *vals)

    result = lax.fori_loop(
        0,
        message_len,
        pop_one_symbol,
        (
            array_to_craymessage(compressed_message, message_shape, tail_limit),
            message,
            cdf_state,
        ),
    )

    return craymessage_to_array(result[0]), result[1], result[2]


def encode(
    messages: ndarray,
    codec: CrayCodec,
    tail_capacity: Optional[int] = None,
    start_buffers: Optional[Sequence[ndarray]] = None,
    tail_capacity_mult: float = 1.5,
) -> Tuple[Sequence[ndarray], Sequence[ndarray]]:
    """
    rANS encoder using craystack API.

    This function encodes a batched array of messages into bytes via batched
    and interleaved entropy coding with the rANS algorithm. This function can
    also be used to add information to an existing compression buffer.

    * ``messages`` should be a JAX-Numpy array of shape
      ``(batch_size, message_length, interleave_level)``. The computation will
      be parallelized over both batches and interleave levels. Each message
      index of the ``interleave_level`` dimension will be coded into the same
      batch output dimension - i.e., the result is a ``batch_size``-length list
      of arrays.

    This rANS variant is based on Craystack, which is described in `A tutorial
    on the range variant of asymmetric numeral systems (J. Townsend)
    <https://arxiv.org/abs/2001.09186>`_.

    The interleaved functionaliy is described in `Interleaved entropy coders
    (F. Giesen) <https://arxiv.org/abs/1402.3392>`_.

    Args:
        messages: A JAX array of messages to be encoded of shape
            ``(batch_size, message_length, interleave_level)``. Note that
            messages in the ``interleave_level`` dimension will be encoded into
            the same batch index.
        codec: A named tuple object containing functions for push and pop
            operations, as well as an initial state fo the CDF functions (for
            context-adaptive coding) and a data type specification for the
            message.
        tail_capacity: An optional manual specification for size of the stack
            tail.
        start_buffers: If provided, the function will append to these existing
            compressed buffers rather than creating a new buffer.
        tail_capacity_mult: A multiplication factor used for calculating the
            size of the tail based on the input message size. Compression
            will silently fail if this value is too small. However, larger
            values imply in longer compression/decompression times.

    Returns:
        A 2-tuple containing:
            A list of arrays of compressed data, one for each batch element.
            The final CDF state.
    """
    if tail_capacity is None:
        tail_capacity = int(tail_capacity_mult * messages[0].size)

    if start_buffers is None:
        message_buffers = jnp.stack(
            [
                craymessage_to_array(empty_message(messages.shape[2:], tail_capacity))[
                    0
                ]
                for _ in range(messages.shape[0])
            ]
        )
        tail_limits = jnp.stack([tail_capacity for _ in range(messages.shape[0])])
    else:
        message_buffers, tail_limits = convert_to_embedded(
            start_buffers, messages.shape[2:], tail_capacity
        )
        tail_limits -= 1  # make sure we don't overwrite info

    def vmap_fun(message, message_buffer, tail_limit, init_cdf_state):
        return _jax_encode(
            message,
            array_to_craymessage(message_buffer, messages.shape[2:], tail_limit),
            codec,
            init_cdf_state,
        )

    map_fun = jax.vmap(vmap_fun)

    (compressed_messages, lims), cdf_state = map_fun(
        messages, message_buffers, tail_limits, codec.cdf_state
    )

    head_size = int(jnp.prod(jnp.array(messages.shape[2:])) * 4)
    array_list = []
    for i, (cm, lim) in enumerate(zip(compressed_messages, lims)):
        if lim - head_size < 0:
            raise RuntimeError(
                f"tail_capacity at batch idx {i} not sufficient for coding."
            )

        splits = jnp.split(cm, [head_size, lim - 1])
        array_list.append(jnp.concatenate((splits[0], splits[2])))

    return array_list, cdf_state


def decode(
    compressed_bytes: Sequence[ndarray],
    message_len: int,
    message_shape: Sequence[int],
    codec: CrayCodec,
    tail_capacity: Optional[int] = None,
    tail_capacity_mult: float = 1.5,
) -> Tuple[ndarray, Sequence[ndarray], Sequence[ndarray]]:
    """
    rANS decoder using craystack API.

    This function decodes a batched array of messages into bytes via batched
    and interleaved entropy coding with the rANS algorithm. The function
    returns the compressed buffer at the end to allow the removal of subsequent
    messages.

    * ``compressed_bytes`` should be a list of ndarrays representing compressed
      data.
    * ``message_shape`` is a tuple specifying the shape of the decoded output.
      For example, if you compressed a ``(batch_size, message_len,
      interleave_level)`` input, then ``message_shape`` should be a 1-tuple
      containing ``interleave_level``.

    This rANS variant is based on craystack, which is described in `A tutorial
    on the range variant of asymmetric numeral systems (J. Townsend)
    <https://arxiv.org/abs/2001.09186>`_.

    The interleaved functionaliy is described in `Interleaved entropy coders
    (F. Giesen) <https://arxiv.org/abs/1402.3392>`_.

    Args:
        compressed_bytes: A list of arrays with compressed messages of length
            ``batch_size``.
        message_len: Length of message to be decoded.
        message_shape: Size of message - must include interleaved level. E.g.,
            if you encoded a message of size ``(5, 100, 2)``, this should be
            ``(2,)``.
        codec: A named tuple object containing functions for push and pop
            operations, as well as an initial state fo the CDF functions (for
            context-adaptive coding) and a data type specification for the
            message.
        tail_capacity: An optional manual specification for size of the stack
            tail.
        start_buffers: If provided, the function will append to these existing
            compressed buffers rather than creating a new buffer.
        tail_capacity_mult: A multiplication factor used for calcualting the
            size of the tail based on the input message size.

    Returns:
        A 3-tuple containing:
            The decoded messages of size
                ``(batch_size, message_len, *message_shape)``.
            A list of arrays of compressed data, one for each batch element
                (after removing the target message).
            The final CDF state.
    """
    if tail_capacity is None:
        tail_capacity = int(
            tail_capacity_mult * message_len * jnp.prod(jnp.array(message_shape))
        )

    compressed_messages, tail_limits = convert_to_embedded(
        compressed_bytes, message_shape, tail_capacity
    )

    def vmap_fun(compressed_message, tail_limit, init_cdf_state):
        return _jax_decode(
            compressed_message,
            tail_limit,
            message_len,
            message_shape,
            codec,
            init_cdf_state,
        )

    map_fun = jax.vmap(vmap_fun)

    (compressed_messages, lims), messages, cdf_state = map_fun(
        compressed_messages, tail_limits, codec.cdf_state
    )
    lims += 1  # correction factor for avoiding encode overwrites

    head_size = int(jnp.prod(jnp.array(message_shape)) * 4)
    array_list = []
    for cm, lim in zip(compressed_messages, lims):
        splits = jnp.split(cm, [head_size, lim - 1])
        array_list.append(jnp.concatenate((splits[0], splits[2])))

    return messages, array_list, cdf_state
