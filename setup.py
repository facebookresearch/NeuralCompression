"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import os
import re

from setuptools import find_packages, setup

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

# alphabetical order
install_requires = [
    "compressai>=1.1.4",
    "jax>=0.2.12",
    "jaxlib>=0.1.65",
    "lpips>=0.1.3",
    "torch>=1.8.1",
    "torchmetrics>=0.3.2",
    "torchvision>=0.9.1",
    "tqdm>=4.61.0",
    "torchmetrics>=0.3.2",
]

setup(
    name="neuralcompression",
    version=version,
    description="A collection of tools for neural compression enthusiasts.",
    long_description_content_type="text/markdown",
    long_description=readme,
    author="Facebook AI Research",
    author_email="mmuckley@fb.com",
    license="MIT",
    project_urls={
        "Source": "https://github.com/facebookresearch/NeuralCompression",
    },
    python_requires=">=3.6",
    setup_requires=["wheel"],
    install_requires=install_requires,
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
)
