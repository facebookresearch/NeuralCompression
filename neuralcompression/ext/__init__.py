# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from torch.utils.cpp_extension import load

print(str(Path(__file__).resolve().parent / "pmf_to_quantized_cdf_py.cc"))
_pmf_to_quantized_cdf = load(
    name="_pmf_to_quantized_cdf",
    sources=[str(Path(__file__).resolve().parent / "pmf_to_quantized_cdf_py.cc")],
    verbose=True,
)

from _pmf_to_quantized_cdf import pmf_to_quantized_cdf  # noqa: E402
