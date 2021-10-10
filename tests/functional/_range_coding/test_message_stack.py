"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from torch import all, int32, int64, ones, randint, tensor
from torch.testing import assert_close

from neuralcompression.functional._range_coding._message_stack import (
    _empty_message_stack,
    _message_stack_to_message,
    _message_to_message_stack,
    _partition_message_stack,
    _pop_from_message_stack,
    _push_to_message_stack,
)


def test_empty_message_stack():
    maximum = 1 << 31

    actual_message, actual_message_stack = _empty_message_stack((1, 2))

    expected_message = tensor([[maximum, maximum]])

    assert_close(actual_message, expected_message)

    assert actual_message_stack == ()

    actual_message, actual_message_stack = _empty_message_stack((2, 2))

    expected_message = tensor(
        [
            [maximum, maximum],
            [maximum, maximum],
        ]
    )

    assert_close(actual_message, expected_message)

    assert actual_message_stack == ()


def test_message_stack_to_message():
    message_stack_a = _empty_message_stack((7, 3))

    for bit in randint(1 << 12, size=(100,) + (7, 3)):
        message_stack_a = _push_to_message_stack(
            message_stack_a,
            bit,
            ones((7, 3), dtype=int32),
            12,
        )

    message_a = _message_stack_to_message(message_stack_a)

    assert message_a.dtype is int64

    message_stack_b = _message_to_message_stack(message_a, (7, 3))

    assert_close(message_stack_a[0], message_stack_b[0])

    message_b = _message_stack_to_message(message_stack_b)

    assert_close(message_a, message_b)


def test_message_to_message_stack():
    message_stack = _empty_message_stack((8, 7))

    starting_indicies = randint(0, 256, size=(1000,) + (8, 7))

    frequencies = randint(1, 256, size=(1000,) + (8, 7)) % (256 - starting_indicies)

    frequencies[frequencies == 0] = 1

    for start, frequency in zip(starting_indicies, frequencies):
        message_stack = _push_to_message_stack(
            message_stack,
            start,
            frequency,
            8,
        )

    encoded = _message_stack_to_message(message_stack)

    assert encoded.dtype == int64

    message_stack = _message_to_message_stack(encoded, (8, 7))

    for start, frequency in reversed(list(zip(starting_indicies, frequencies))):
        cf, pop = _pop_from_message_stack(message_stack, 8)

        assert all(start <= cf) and all(cf < start + frequency)

        message_stack = pop(start, frequency)

    assert all(message_stack[0] == _empty_message_stack((8, 7))[0])


def test_partition_message_stack():
    maximum = 1 << 31

    message, _ = _empty_message_stack((1, 2))

    message_stack = message, _empty_message_stack((1, 2))

    actual_message, actual_message_stack = _partition_message_stack(
        message_stack,
        n=2,
    )

    expected_message = tensor([[maximum, maximum], [maximum, maximum]])

    assert_close(actual_message, expected_message)

    assert actual_message_stack == ()


def test_pop_from_message_stack():
    message_stack = _empty_message_stack((1, 2))

    actual_interval_starting_indicies, _ = _pop_from_message_stack(
        message_stack,
        precision=16,
    )

    expected_interval_starting_indicies = tensor([[0, 0]])

    assert_close(
        actual_interval_starting_indicies,
        expected_interval_starting_indicies,
    )


def test_push_to_message_stack():
    quantized = 140737488355328

    message_stack = _empty_message_stack((1, 2))

    actual_message, actual_message_stack = _push_to_message_stack(
        message_stack,
        tensor([[0, 0]]),
        tensor([[1, 1]]),
        precision=16,
    )

    expected_message = tensor([[quantized, quantized]])

    assert_close(
        actual_message,
        expected_message,
    )

    assert actual_message_stack == ()
