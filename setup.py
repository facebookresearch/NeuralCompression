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

# from https://github.com/facebookresearch/ClassyVision/blob/master/setup.py
# get version string from module
with open(
    os.path.join(os.path.dirname(__file__), "neuralcompression/__init__.py"), "r"
) as f:
    readval = re.search(r"__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if readval is None:
        raise RuntimeError("Version not found.")
    version = readval.group(1)
    print("-- Building version " + version)

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
