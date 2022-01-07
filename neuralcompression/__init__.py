# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("neuralcompression")
except PackageNotFoundError:
    # package is not installed
    import warnings

    warnings.warn("Could not retrieve neuralcompression version!")
