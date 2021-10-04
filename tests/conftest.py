"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import PIL
import numpy
import torch


def create_random_image(file_path, shape):
    img = numpy.random.random_sample(shape)
    img = (255.0 / img.max() * (img - img.min())).astype(numpy.uint8)
    res = torch.tensor(img).to(torch.float) / 255
    img = numpy.transpose(img, (1, 2, 0))
    img = PIL.Image.fromarray(img)
    img.save(file_path)
    return res
