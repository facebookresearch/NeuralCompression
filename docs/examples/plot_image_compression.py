# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Image compression
=================
"""

import matplotlib.pyplot
import numpy
import torch

from neuralcompression.models import FactorizedPriorAutoencoder
import skimage.data
import skimage.transform

image = skimage.data.cat()

image = skimage.transform.resize(image, (256, 256))

input_shape = image.shape

r, c, channels = image.shape

matplotlib.pyplot.imshow(image)

matplotlib.pyplot.show()

# %%
# Using an image compression model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The neuralcompression.models module provides a variety of popular image
# compression models. For example,  FactorizedPrior, ScaleHyperprior, and
# MeanScaleHyperprior models from Belle, et. al.
#
# Quality
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# A model’s compression quality is characterized by model-specific constants.
# For example, for the aforementioned auto-encoders, quality is described by
# the number of compression filters, the number of network filters, and the
# distortion trade-off. Because other models may use different constants, I
# recommend reading the appropriate model documentation.
#
# Here, I use the default quality arguments when instantiating the
# FactorizedPrior model (128 compression filters, 192 network filters, and a
# distortion trade-off of 0.01):

network = FactorizedPriorAutoencoder(128, 192)

# %%
# Pre-training
# ^^^^^^^^^^^^
#
# NeuralCompression includes pre-trained weights for popular quality values.
# For the FactorizedPrior model, the following weights are available:
#
# For other models, see the appropriate model documentation. Each model’s
# documentation includes a rate-distortion curve to illustrate the performance
# of the weights.

url = "https://dl.fbaipublicfiles.com/neuralcompression/models/factorized_prior_vimeo_90k_mse_128_192_0_025.pth"

state_dict = torch.hub.load_state_dict_from_url(url)

network.load_state_dict(state_dict, strict=False)

image = numpy.reshape(image, (channels, r, c))

image = numpy.expand_dims(image, 0)

x = torch.tensor(image).to(torch.float32)

# %%
# Compress an image to a bit-stream
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Unlike many PyTorch models for computer vision tasks, the module’s forward
# method isn’t used for inference purposes. Instead, each model exposes
# specialized compress and decompress methods written to perform these
# inference tasks. The signatures of these methods may vary between models so
# reading the appropriate model documentation is recommended. Typically,
# however, the compress method takes an image (a Tensor) and returns a list of
# bit-strings.

with torch.no_grad():
    strings, broadcast_size = network.compress(x)

# %%
# Decompress a bit-stream to an image
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

with torch.no_grad():
    x_hat = network.decompress(strings, broadcast_size)

x_hat = x_hat.squeeze()

x_hat = torch.reshape(x_hat, input_shape)

matplotlib.pyplot.imshow(x_hat)

matplotlib.pyplot.show()
