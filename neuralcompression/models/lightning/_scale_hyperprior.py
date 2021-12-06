# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.hub

import neuralcompression.models
from ._prior_autoencoder import PriorAutoencoder


class ScaleHyperpriorAutoencoder(PriorAutoencoder):
    network: neuralcompression.models.ScaleHyperpriorAutoencoder

    def __init__(
        self,
        network_channels: int = 128,
        compression_channels: int = 192,
        in_channels: int = 3,
        distortion_trade_off: float = 1e-2,
        optimizer_lr: float = 1e-3,
        bottleneck_optimizer_lr: float = 1e-3,
        pretrained: bool = False,
    ):
        super(ScaleHyperpriorAutoencoder, self).__init__(
            distortion_trade_off,
            optimizer_lr,
            bottleneck_optimizer_lr,
        )

        self.network = neuralcompression.models.ScaleHyperpriorAutoencoder(
            network_channels,
            compression_channels,
            in_channels,
        )

        if pretrained:
            url = (
                "https://dl.fbaipublicfiles.com"
                + "/"
                + "neuralcompression"
                + "/"
                + "models"
                + "/"
                + "scale_hyperprior"
                + "_"
                + "vimeo_90k"
                + "_"
                + "mse"
                + "_"
                + str(network_channels)
                + "_"
                + str(compression_channels)
                + "_"
                + str(distortion_trade_off).replace(".", "_")
                + ".pth"
            )

            state_dict = torch.hub.load_state_dict_from_url(url)

            self.network.load_state_dict(state_dict)
