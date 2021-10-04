"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import typing

import numpy  # type: ignore

_Schedule = typing.Dict[str, typing.Union[typing.List[float], typing.List[int]]]


def _get_scheduled_parameters(
    parameter: float, schedule: _Schedule, step: int, ignore_schedule: bool = False
) -> float:
    if ignore_schedule:
        return parameter

    index = numpy.where(step < numpy.array(schedule["t"] + [step + 1]))

    return parameter * schedule["x"][index[0][0]]


def _weighted_rate_loss(
    nbpp: float,
    qbpp: float,
    step_counter: int,
    ignore_schedule: bool = False,
    lambda_b: float = 0.0625,
    lambda_rates: typing.Optional[typing.Dict[str, float]] = None,
    lambda_schedule: typing.Optional[_Schedule] = None,
    regime: str = "a",
    target_rates: typing.Optional[typing.Dict[str, float]] = None,
    target_schedule: typing.Optional[_Schedule] = None,
) -> typing.Tuple[float, float]:
    if not lambda_rates:
        lambda_rates = {
            "a": 2.0,
            "b": 1.0,
            "c": 0.5,
        }

    if not lambda_schedule:
        lambda_schedule = {
            "t": [50000],
            "x": [2.0, 1.0],
        }

    if not target_rates:
        target_rates = {
            "a": 0.14,
            "b": 0.30,
            "c": 0.45,
        }

    if not target_schedule:
        target_schedule = {
            "t": [50000],
            "x": [1.4, 1.0],
        }

    lambda_a = _get_scheduled_parameters(
        lambda_rates[regime],
        lambda_schedule,
        step_counter,
        ignore_schedule,
    )

    lambda_b = _get_scheduled_parameters(
        lambda_b,
        lambda_schedule,
        step_counter,
        ignore_schedule,
    )

    assert lambda_a > lambda_b

    target = _get_scheduled_parameters(
        target_rates[regime],
        target_schedule,
        step_counter,
        ignore_schedule,
    )

    if qbpp > target:
        rate_penalty = lambda_a
    else:
        rate_penalty = lambda_b

    return rate_penalty * nbpp, rate_penalty
