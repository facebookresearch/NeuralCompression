# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle
from io import BytesIO
from typing import Any


def pickle_size_of(obj: Any) -> int:
    """
    Measure the size of a pickle file containing obj.

    This function can be used to assess the size of obj if it were written
    to disk. No actual disk-writing is included: the process is simulated
    using a BytesIO object. This is useful for assessing the size of a
    compressed image including associated metadata.

    Args:
        obj: The object to be written to a file.

    Returns:
        The size of the object if it were written to a pickle file.
    """
    buffer = BytesIO()
    pickle.dump(obj, buffer)

    num_bytes = buffer.getbuffer().nbytes

    buffer.close()

    return num_bytes
