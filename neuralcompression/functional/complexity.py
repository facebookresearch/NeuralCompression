"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from copy import copy
from typing import Any, Dict, Optional, Sequence, Tuple

from fvcore.nn import FlopCountAnalysis
from fvcore.nn.jit_handles import Handle, generic_activation_jit
from torch import nn

# These ops aren't supported by fvcore - we estimate their FLOPs
# by counting 1 FLOP per element of the output tensor. Note that
# this method handles broadcasting input shapes.
_OPS_TO_ADD = [
    f"aten::{name}"
    for name in [
        "abs",
        "add",
        "div",
        "erfc",
        "exp",
        "log",
        "log10",
        "log2",
        "mul",
        "neg",
        "pow",
        "reciprocal",
        "rsqrt",
        "rsub",
        "sigmoid",
        "sign",
        "softplus",
        "sqrt",
        "sub",
        "tanh",
        "uniform",
    ]
]

# Creating counter functions for all the ops listed above, in addition
# to their inplace (trailing underscore) equivalents.
_ADDITONAL_COUNTERS = {
    name: generic_activation_jit(name)
    for name in _OPS_TO_ADD + [f"{n}_" for n in _OPS_TO_ADD]
}

# Expanding on the 0-FLOP op list from fvcore here:
# https://github.com/facebookresearch/fvcore/blob/166a030e093013a934642ca3744592a2e3de5ea2/fvcore/nn/jit_analysis.py#L27
_OPS_TO_IGNORE = ["aten::empty_like"]


def count_flops(
    module: nn.Module,
    inputs: Sequence[Any],
    counter_overrides: Optional[Dict[str, Handle]] = None,
) -> Tuple[float, Dict[str, float], Dict[str, int]]:
    """
    Counts the FLOPs in the forward function of an ``nn.Module``.

    Given a model, this counter first uses TorchScript to trace the
    model's forward function into a series of linear algebra operations
    (e.g. matmuls, convolutions, etc.). Then the FLOPs of each operation
    are counted, provided that the operator type has a corresponding flop
    counting function registered. Note that a multiply-accumulate is
    considered one FLOP, not two.

    If an operator does not have a registered counter, it is ignored in the
    flop count and is logged as an unregistered op in the function's return
    value (see below). Most common operations already have
    registered counters, but it may be necessary to register additional
    operators in the counter_overrides field.

    This function relies on the fvcore flop counter, accessible here:
    https://github.com/facebookresearch/fvcore

    Args:
        module: the ``nn.Module`` whose forward function will be profiled.
        inputs: a tuple of Tensors to pass as inputs to ``module``
            when it is traced.
        counter_overrides: a dictionary mapping operator names to
            FLOP-counting functions, to extend/override the default
            registered operators. A counting function's signature
            should take in a list of operator inputs and a list
            of operator outputs (in that order), and return the
            operator's FLOP count. See the fvcore documentation
            for more details.

    Returns:
        A tuple of:
            - The total number of FLOPs recorded in the model's
            forward function (returned as a float).
            - A dictionary breaking down the total model FLOPs by
            operator (i.e. a dictionary mapping from operator names
            to the total FLOPs performed by all calls to that operator
            in the model).
            - A dictionary recording all the unsupported model operations,
            i.e. operations who don't have associated FLOP-counting
            functions. This dictionary maps the names of unsupported operators
            to the number of times that operator was invoked by the model.
    """

    # fvcore requires a tuple of inputs and
    # will often crash if a list is passed
    inputs = tuple(inputs)

    counters_to_add = copy(_ADDITONAL_COUNTERS)
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
