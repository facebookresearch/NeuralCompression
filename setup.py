"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import re
from pathlib import Path

import setuptools
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


setuptools.setup(
    # version=version,
    project_urls={
        "Source": "https://github.com/facebookresearch/NeuralCompression",
    },
    packages=setuptools.find_packages(
        exclude=[
            "tests",
            "projects",
        ]
    ),
    ext_modules=[
        CppExtension(
            "neuralcompression.ext._foo",
            [
                str(
                    Path(__file__).resolve().parent
                    / "neuralcompression"
                    / "ext"
                    / "foo_py.cc"
                )
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    use_scm_version={"write_to": "neuralcompression/_version.py"},
)
