# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy
import numpy as np
import PIL
import torch


def create_input(shape):
    x = np.arange(np.product(shape)).reshape(shape)

    return torch.from_numpy(x).to(torch.get_default_dtype())


def create_random_image(file_path, shape):
    img = numpy.random.random_sample(shape)
    img = (255.0 / img.max() * (img - img.min())).astype(numpy.uint8)
    res = torch.tensor(img).to(torch.float) / 255
    img = numpy.transpose(img, (1, 2, 0))
    img = PIL.Image.fromarray(img)
    img.save(file_path)
    return res


def rand_im(shape, rng=None):
    if rng is None:
        rng = numpy.random.default_rng()
    return torch.tensor(rng.uniform(size=shape), dtype=torch.get_default_dtype())
