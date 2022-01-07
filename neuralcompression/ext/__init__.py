# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from ._extension_loader import load_extension

# load pmf_to_quantized_cdf
_extension_folder = Path(__file__).resolve().parent
_extension_name = "_pmf_to_quantized_cdf"
_extension_files = ["pmf_to_quantized_cdf_py.cc"]

_pmf_to_quantized_cdf = load_extension(
    _extension_name, _extension_folder, _extension_files
)

from _pmf_to_quantized_cdf import pmf_to_quantized_cdf  # noqa: E402
