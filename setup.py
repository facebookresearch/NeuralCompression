"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import re
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
    cmdclass={"build_ext": BuildExtension},
    ext_modules=[
        CppExtension(
            "neuralcompression.ext._pmf_to_quantized_cdf",
            [
                str(
                    Path(__file__).resolve().parent
                    / "neuralcompression"
                    / "ext"
                    / "pmf_to_quantized_cdf_py.cc"
                )
            ],
        ),
    ],
    version=version,
)
