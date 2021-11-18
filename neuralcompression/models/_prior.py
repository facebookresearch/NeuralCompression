# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Set

import torch
from pytorch_lightning import LightningModule
from torch.nn import Parameter
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import neuralcompression.layers


class Prior(LightningModule):
    """A base class for implementing neural compression autoencoders.

    The class couples a ``bottleneck_module`` (e.g. the ``EntropyBottleneck``
    module provided by the CompressionAI package) with an autoencoder
    (i.e. ``encoder`` and ``decoder``).

    Using the base class is as straightforward as inheriting from the class and
    defining an ``encoder_module`` and ``decoder_module``. You may optionally
    provide a ``hyper_encoder_module`` and ``hyper_decoder_module`` (e.g. for
    implementing Hyperprior architectures).

    The ``neuralcompression.layers`` package includes a standard encoder
    (``AnalysisTransformation2D``), decoder (``SynthesisTransformation2D``),
    hyper encoder (``HyperAnalysisTransformation2D``), and hyper decoder
    (``HyperSynthesisTransformation2D``).

    Args:
        architecture: an autoencoder with an encoder and decoder modules (e.g.
            the ``FactorizedPrior``, ``ScaleHyperprior``, or
            ``MeanScaleHyperprior`` modules from ``neuralcompression.models``).
        bottleneck_optimizer_lr: the learning rate for the bottleneck module.
        optimizer_lr: the learning rate for the autoencoder.
    """

    def __init__(
        self,
        architecture: neuralcompression.layers.Prior,
        bottleneck_optimizer_lr: float = 1e-4,
        optimizer_lr: float = 1e-3,
    ):
        super(Prior, self).__init__()

        self.autoencoder = architecture

        self.bottleneck_optimizer_lr = bottleneck_optimizer_lr

        self.optimizer_lr = optimizer_lr

        self.save_hyperparameters()

        self.example_input_array = torch.zeros(2, 3, 256, 256)

    def _autoencoder_parameters(self) -> Set[str]:
        autoencoder_parameters = set()

        for name, parameter in self.named_parameters():
            if not name.endswith(".quantiles") and parameter.requires_grad:
                autoencoder_parameters.add(name)

        return autoencoder_parameters

    def _bottleneck_optimizer_parameters(self) -> List[Parameter]:
        bottleneck_optimizer_parameters = []

        for bottleneck_parameter in sorted(self._bottleneck_parameters()):
            bottleneck_optimizer_parameter = self._parameters_dict()[
                bottleneck_parameter
            ]

            bottleneck_optimizer_parameters += [bottleneck_optimizer_parameter]

        return bottleneck_optimizer_parameters

    def _bottleneck_parameters(self) -> Set[str]:
        bottleneck_parameters = set()

        for name, parameter in self.named_parameters():
            if name.endswith(".quantiles") and parameter.requires_grad:
                bottleneck_parameters.add(name)

        return bottleneck_parameters

    def _intersection_parameters(self) -> Set[str]:
        return self._autoencoder_parameters() & self._bottleneck_parameters()

    def _optimizer_parameters(self) -> List[Parameter]:
        optimizer_parameters = []

        for parameter in sorted(self._autoencoder_parameters()):
            optimizer_parameters += [self._parameters_dict()[parameter]]

        return optimizer_parameters

    def _parameters_dict(self) -> Dict[str, Parameter]:
        return dict(self.named_parameters())

    def _union_parameters(self) -> Set[str]:
        return self._autoencoder_parameters() | self._bottleneck_parameters()

    def configure_optimizers(self):
        assert len(self._intersection_parameters()) == 0

        assert len(self._union_parameters()) - len(self._parameters_dict().keys()) == 0

        bottleneck_optimizer = Adam(
            self._bottleneck_optimizer_parameters(),
            lr=self.bottleneck_optimizer_lr,
        )

        optimizer = self.optimizer(
            self._optimizer_parameters,
            lr=self.optimizer_lr,
        )

        bottleneck_lr_scheduler = Adam(
            self._bottleneck_optimizer_parameters(),
            lr=self.bottleneck_optimizer_lr,
        )

        lr_scheduler = ReduceLROnPlateau(optimizer, "min")

        return [
            {
                "lr_scheduler": bottleneck_lr_scheduler,
                "monitor": "val_loss",
                "optimizer": bottleneck_optimizer,
            },
            {
                "lr_scheduler": lr_scheduler,
                "monitor": "val_loss",
                "optimizer": optimizer,
            },
        ]
