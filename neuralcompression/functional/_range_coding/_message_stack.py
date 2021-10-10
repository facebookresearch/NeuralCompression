"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Callable, Optional, Tuple

from torch import Tensor, any, cat, div, full, int64, ravel, remainder, sum, tensor

_MessageStack = Tuple[Tensor, Optional["_MessageStack"]]


def _empty_message_stack(shape: Tuple[int, ...]) -> _MessageStack:
    """

    Args:
        shape:

    Returns:
        an empty ``_MessageStack``.
    """
    return full(shape, 1 << 31), ()


def _message_stack_to_message(message_stack: _MessageStack) -> Tensor:
    """Transforms a ``_MessageStack`` into a message (i.e. a
        ``pytorch.Tensor``).

    Args:
        message_stack:

    Returns:
    """
    message, message_stack = message_stack

    message = ravel(message)

    messages = [message >> 32, message]

    while message_stack:
        message, message_stack = message_stack

        messages += [message]

    return cat(messages)


def _message_to_message_stack(message: Tensor) -> _MessageStack:
    """Transforms a message (i.e. a ``pytorch.Tensor``) into a
        ``_MessageStack``.

    Args:
        message:

    Returns:
    """
    return message[0] << 32 | message[1], (message[2:], ())


def _partition_message_stack(
    message_stack: _MessageStack,
    n: int = 0,
) -> _MessageStack:
    """
    Args:
        message_stack:
        n:

    Returns:
    """
    if n == 0:
        return message_stack

    messages = []

    while n > 0:
        message, message_stack = message_stack

        if n >= len(message):
            messages += [message]

            n -= len(message)
        else:
            messages += [message[:n]]

            message_stack = message[n:], message_stack

            break

    return cat(messages), message_stack


def _pop_from_message_stack(
    message_stack: _MessageStack,
    precision: int,
) -> Tuple[Tensor, Callable[[Tensor, Tensor], _MessageStack]]:
    """
    Args:
        message_stack:
        precision:

    Returns:
    """
    previous_message, previous_message_stack = message_stack

    previous_message = previous_message.type(int64)

    interval_starting_indicies = previous_message & (1 << precision) - 1

    def f(starting_indicies: Tensor, frequencies: Tensor) -> _MessageStack:
        messages = (
            frequencies * (previous_message >> precision)
            + interval_starting_indicies
            - starting_indicies
        )

        indicies = messages < (1 << 31)

        n = sum(indicies)

        if n > 0:
            next_message, next_message_stack = _partition_message_stack(
                previous_message_stack, int(n)
            )

            try:
                messages[indicies] = tensor(messages[indicies] << 32) | next_message
            except TypeError:
                messages = tensor(messages << 32) | next_message
        else:
            next_message_stack = previous_message_stack

        return messages, next_message_stack

    return interval_starting_indicies, f


def _push_to_message_stack(
    message_stack: _MessageStack,
    starting_indicies: Tensor,
    frequencies: Tensor,
    precision: int,
) -> _MessageStack:
    """
    Args:
        message_stack:
        starting_indicies:
        frequencies:
        precision:

    Returns:
    """
    message, message_stack = message_stack

    assert message.shape == starting_indicies.shape == frequencies.shape

    indicies = message >= (((1 << 31) >> precision) << 32) * frequencies

    if any(indicies) > 0:
        message_stack = message[indicies], message_stack

        message[indicies] >>= 32

    quotients = div(message, frequencies, rounding_mode="floor")

    remainders = remainder(message, frequencies)

    return (quotients << precision) + remainders + starting_indicies, message_stack
