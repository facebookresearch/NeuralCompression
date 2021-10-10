"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from torch import tensor
from torch.testing import assert_close

from neuralcompression.functional._range_coding._message_stack import (
    _empty_message_stack,
    _message_stack_to_message,
    _message_to_message_stack,
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
    maximum = 1 << 31

    message_stack = _empty_message_stack((1, 2))

    actual_message = _message_stack_to_message(message_stack)

    expected_message = tensor([0, 0, maximum, maximum])

    assert_close(actual_message, expected_message)


def test_message_to_message_stack():
    maximum = 1 << 31

    message = tensor([0, 0, maximum, maximum])

    actual_message_stack = _message_to_message_stack(message)

    actual_message_stack_message, _ = actual_message_stack

    expected_message = tensor(0)

    assert_close(actual_message_stack_message, expected_message)

    _, actual_message_stack = actual_message_stack

    actual_message_stack_message, _ = actual_message_stack

    expected_message = tensor([maximum, maximum])

    assert_close(actual_message_stack_message, expected_message)

    _, actual_message_stack = actual_message_stack

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


def test_slice_message_stack():
    assert True
