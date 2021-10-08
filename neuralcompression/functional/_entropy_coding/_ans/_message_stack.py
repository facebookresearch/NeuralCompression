from typing import Callable, Optional, Tuple

from torch import Tensor, any, cat, div, int64, remainder, sum, tensor

_MessageStack = Tuple[Tensor, Optional["_MessageStack"]]


def _pop(
    message_stack: _MessageStack,
    precision: int,
) -> Tuple[Tensor, Callable[[Tensor, Tensor], _MessageStack]]:
    previous_messages, previous_stack = message_stack

    previous_messages = previous_messages.type(int64)

    interval_starting_indicies = previous_messages & (1 << precision) - 1

    def f(starting_indicies: Tensor, frequencies: Tensor) -> _MessageStack:
        messages = (
            frequencies * (previous_messages >> precision)
            + interval_starting_indicies
            - starting_indicies
        )

        indicies = messages < (1 << 31)

        n = sum(indicies)

        if n > 0:
            next_messages, stack = _slice(previous_stack, int(n))

            try:
                messages[indicies] = tensor(messages[indicies] << 32) | next_messages
            except TypeError:
                messages = tensor(messages << 32) | next_messages
        else:
            stack = previous_stack

        return messages, stack

    return interval_starting_indicies, f


def _push(
    message_stack: _MessageStack,
    starting_indicies: Tensor,
    frequencies: Tensor,
    precision: int,
) -> _MessageStack:
    message, stack = message_stack

    assert message.shape == starting_indicies.shape == frequencies.shape

    indicies = message >= (((1 << 31) >> precision) << 32) * frequencies

    if any(indicies) > 0:
        stack = message[indicies], stack

        message[indicies] >>= 32

    quotients = div(message, frequencies, rounding_mode="floor")

    remainders = remainder(message, frequencies)

    return (quotients << precision) + remainders + starting_indicies, stack


def _slice(
    stack: _MessageStack,
    n: int = 0,
) -> _MessageStack:
    if n == 0:
        return stack

    messages = []

    while n > 0:
        message, stack = stack

        if n >= len(message):
            messages += [message]

            n -= len(message)
        else:
            messages += [message[:n]]

            stack = message[n:], stack

            break

    return cat(messages), stack
