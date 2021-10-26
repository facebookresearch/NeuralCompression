"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import os
import re
from pathlib import Path

from setuptools import find_packages, setup
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

with open("README.md", encoding="utf8") as f:
    readme = f.read()

setup(
    name="neuralcompression",
    version=version,
    description="A collection of tools for neural compression enthusiasts.",
    long_description_content_type="text/markdown",
    long_description=readme,
    author="Facebook AI Research",
    license="MIT",
    project_urls={
        "Source": "https://github.com/facebookresearch/NeuralCompression",
    },
    python_requires=">=3.6",
    setup_requires=["wheel"],
    packages=find_packages(
        exclude=[
            "tests",
            "projects",
        ]
    ),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Archiving :: Compression",
    ],
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
    cmdclass={"build_ext": BuildExtension},
)
