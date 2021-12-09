# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import copy
from typing import Any, Dict, Optional, Sequence, Tuple

from fvcore.nn import FlopCountAnalysis
from fvcore.nn.jit_handles import Handle, generic_activation_jit
from torch import nn

# These ops aren't supported (i.e. are ignored) by fvcore - we calculate
# their flops by counting 1 flop per element of the output tensor. Note that
# this approach handles broadcasting input shapes.
_SINGLE_FLOP_OPS_TO_ADD = [
    f"aten::{name}"
    for name in [
        "abs",
        "add",
        "div",
        "mul",
        "neg",
        "rsub",
        "sign",
        "sub",
    ]
]


# These ops are also unsupported by fvcore, but unlike those in the
# above list, these ops likely require more than 1 flop per
# output element, and their counts can vary depending on
# platform implementation. For correctness' sake, by default we keep
# these ops as unregistered/ignored. However, if your model has lots of these
# operations and you wish to obtain a rough estimate of their contributions
# to total model complexity, for convenience we expose the
# use_single_flop_estimates flag. If True, this flag will
# register counter functions for the ops below
# that count 1 flop per output element. If you know exactly
# how many flops some of these ops should have on your platform,
# use the counter_overrides argument.
_SINGLE_FLOP_ESTIMATES_TO_ADD = [
    f"aten::{name}"
    for name in [
        "erfc",
        "exp",
        "log",
        "log10",
        "log2",
        "pow",
        "reciprocal",
        "rsqrt",
        "sigmoid",
        "softplus",
        "sqrt",
        "tanh",
        "uniform",
    ]
]

# Creating counter functions for all the ops listed in
# _SINGLE_FLOP_OPS_TO_ADD, in addition to their inplace
# (trailing underscore) equivalents.
_SINGLE_FLOP_OP_COUNTERS = {
    name: generic_activation_jit(name)
    for name in _SINGLE_FLOP_OPS_TO_ADD + [f"{n}_" for n in _SINGLE_FLOP_OPS_TO_ADD]
}

# Same as the above, but for the ops to estimate.
_SINGLE_FLOP_ESTIMATE_COUNTERS = {
    name: generic_activation_jit(name)
    for name in _SINGLE_FLOP_ESTIMATES_TO_ADD
    + [f"{n}_" for n in _SINGLE_FLOP_ESTIMATES_TO_ADD]
}


# Expanding on the 0-flop op list from fvcore here:
# https://github.com/facebookresearch/fvcore/blob/166a030e093013a934642ca3744592a2e3de5ea2/fvcore/nn/jit_analysis.py#L27
_OPS_TO_IGNORE = ["aten::empty_like"]


def count_flops(
    module: nn.Module,
    inputs: Sequence[Any],
    counter_overrides: Optional[Dict[str, Handle]] = None,
    use_single_flop_estimates: bool = False,
) -> Tuple[float, Dict[str, float], Dict[str, int]]:
    """Counts the flops in the forward function of an ``nn.Module``.

    Given a model, this counter first uses TorchScript to trace the model's
    forward function into a series of linear algebra operations (e.g. matmuls,
    convolutions, etc.). Then the flops of each operation are counted, provided
    that the operator type has a corresponding flop counting function
    registered. Note that a multiply-accumulate is considered one flop, not
    two.

    This function relies on the
    `fvcore flop counter <https://github.com/facebookresearch/fvcore>`_.

    Note:
        If an operator does not have a registered counter, it is ignored in the
        flop count and is logged as an unregistered op in the function's return
        value (see below). Most common operations already have registered
        counters, but it may be necessary to register additional operators in
        the ``counter_overrides`` field.

    Args:
        module: The ``nn.Module`` whose forward function will be profiled.
        inputs: A sequence of variables to pass as inputs to ``module`` when it
            is traced.
        counter_overrides: A dictionary mapping operator names to flop-counting
            functions, to extend/override the default registered operators. A
            counting function's signature should take in a list of operator
            inputs and a list of operator outputs (in that order), and return
            the operator's flop count. See the fvcore documentation for more
            details.
        use_single_flop_estimates: Dictates whether to use approximate
            flop-counting functions for many elementwise ops not supported by
            fvcore. Many ops, like sqrt, log, pow, etc., are by default ignored
            by fvcore, since their true flop count can vary by platform op
            implementation. For correctness' sake, we too ignore these ops by
            default. However, if your model has lots of these types of
            operations and you wish to obtain a rough estimate of their
            contributions to total model complexity, passing this flag as
            ``True`` will register conservative counter estimates for these ops
            that count 1 flop per output tensor element (e.g. 1 sqrt = 1 flop).
            The full list of ops that this flag registers counters for can be
            found in ``_SINGLE_FLOP_ESTIMATES_TO_ADD``. If you know exactly how
            many flops some of these ops should have on your platform, use the
            ``counter_overrides`` argument.

    Returns:
        * The total number of flops recorded in the model's forward
          function (returned as a float).
        * A dictionary breaking down the total model flops by operator
          (i.e. a dictionary mapping from operator names to the total flops
          performed by all calls to that operator in the model).
        * A dictionary recording all the unsupported model operations, i.e.
          operations who don't have associated flop-counting functions.
          This dictionary maps the names of unsupported operators to the
          number of times that operator was invoked by the model.
    """

    # fvcore requires a tuple of inputs and
    # will often crash if a list is passed
    inputs = tuple(inputs)

    counters_to_add = copy(_SINGLE_FLOP_OP_COUNTERS)
    if use_single_flop_estimates:
        counters_to_add.update(_SINGLE_FLOP_ESTIMATE_COUNTERS)
    if counter_overrides is not None:
        counters_to_add.update(counter_overrides)

    counter = FlopCountAnalysis(module, inputs).set_op_handle(**counters_to_add)
    for name in _OPS_TO_IGNORE:
        counter._ignored_ops.add(name)
    return (
        counter.total(),
        dict(counter.by_operator().items()),
        counter.unsupported_ops(),
    )
