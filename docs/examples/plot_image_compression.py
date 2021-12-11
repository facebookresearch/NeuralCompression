# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Image compression
=================
"""

import matplotlib.pyplot
import torch
from PIL import Image
from torchvision.transforms import Compose, ConvertImageDtype, PILToTensor

from neuralcompression.models import FactorizedPriorAutoencoder

x = Image.open("./image.png")

transform = Compose(
    [
        PILToTensor(),
        ConvertImageDtype(torch.float32),
    ]
)

x = transform(x).unsqueeze(0)

matplotlib.pyplot.imshow(x.squeeze().permute(1, 2, 0))

matplotlib.pyplot.tight_layout()

matplotlib.pyplot.show()

# %%
# Model usage
# ~~~~~~~~~~~
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

network = FactorizedPriorAutoencoder()

# %%
# Pre-training
# ^^^^^^^^^^^^
#
# NeuralCompression includes pre-trained weights for popular quality values.
# For the FactorizedPrior model, the following weights are available:
#
# .. list-table::
#    :header-rows: 1
#
#    * - Data
#      - :math:`N`
#      - :math:`M`
#      - Metric
#      - :math:`\lambda`
#    * - Vimeo90kSeptuplet
#      - :math:`128`
#      - :math:`192`
#      - MSE
#      - :math:`0.0015625`
#    * - Vimeo90kSeptuplet
#      - :math:`128`
#      - :math:`192`
#      - MSE
#      - :math:`0.003125`
#    * - Vimeo90kSeptuplet
#      - :math:`128`
#      - :math:`192`
#      - MSE
#      - :math:`0.00625`
#    * - Vimeo90kSeptuplet
#      - :math:`128`
#      - :math:`192`
#      - MSE
#      - :math:`0.0125`
#    * - Vimeo90kSeptuplet
#      - :math:`128`
#      - :math:`192`
#      - MSE
#      - :math:`0.025`
#    * - Vimeo90kSeptuplet
#      - :math:`192`
#      - :math:`320`
#      - MSE
#      - :math:`0.05`
#    * - Vimeo90kSeptuplet
#      - :math:`192`
#      - :math:`320`
#      - MSE
#      - :math:`0.1`
#    * - Vimeo90kSeptuplet
#      - :math:`192`
#      - :math:`320`
#      - MSE
#      - :math:`0.2`
#
# For other models, see the appropriate model documentation. Each model’s
# documentation includes a rate-distortion curve to illustrate the performance
# of the weights.

url = "https://dl.fbaipublicfiles.com/neuralcompression/models/factorized_prior_vimeo_90k_mse_128_192_0_025.pth"

state_dict = torch.hub.load_state_dict_from_url(url)

network.load_state_dict(state_dict, strict=False)

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

matplotlib.pyplot.imshow(x_hat.squeeze().permute(1, 2, 0))

matplotlib.pyplot.tight_layout()

matplotlib.pyplot.show()
