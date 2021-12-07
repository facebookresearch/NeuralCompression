# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing

import numpy  # type: ignore

_Schedule = typing.Dict[str, typing.List[typing.Union[int, float]]]


def _get_scheduled_parameters(
    parameter: float, schedule: _Schedule, step: int, ignore_schedule: bool = False
) -> float:
    if ignore_schedule:
        return parameter

    index = numpy.where(step < numpy.array(schedule["steps"] + [step + 1]))

    return parameter * schedule["parameters"][index[0][0]]


def _weighted_rate_loss(
    a: float,
    b: float,
    schedule: _Schedule,
    target: float,
    target_schedule: _Schedule,
    nbpp: float,
    qbpp: float,
    step: int,
    ignore_schedule: bool = False,
) -> float:
    a = _get_scheduled_parameters(a, schedule, step, ignore_schedule)
    b = _get_scheduled_parameters(b, schedule, step, ignore_schedule)

    target = _get_scheduled_parameters(target, target_schedule, step, ignore_schedule)

    if qbpp > target:
        penalty = a
    else:
        penalty = b

    return penalty * nbpp
