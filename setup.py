"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from pathlib import Path

import setuptools
from torch.utils.cpp_extension import BuildExtension, CppExtension

setuptools.setup(
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
