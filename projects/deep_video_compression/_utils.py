# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, NamedTuple, Optional

import torch
import torch.nn as nn
from torch import Tensor

import neuralcompression.functional as ncF


class LossFunctions(NamedTuple):
    distortion_fn: Callable[[Tensor, Tensor], Tensor]
    entropy_fn: Optional[Callable[[Tensor, int], Tensor]]


class LossValues(NamedTuple):
    distortion_loss: Tensor
    flow_entropy_loss: Optional[Tensor]
    resid_entropy_loss: Optional[Tensor]


class OutputTensors(NamedTuple):
    flow: Optional[Tensor]
    image1: Optional[Tensor]
    image2: Optional[Tensor]
    image2_est: Optional[Tensor]


class DvcStage(nn.Module):
    def collect_parameters(self):
        model_parameters = {
            n: p
            for n, p in self.named_parameters()
            if not n.endswith(".quantiles") and p.requires_grad
        }
        quantile_parameters = {
            n: p
            for n, p in self.named_parameters()
            if n.endswith(".quantiles") and p.requires_grad
        }

        return model_parameters, quantile_parameters

    def forward(self, image1, image2):
        raise NotImplementedError

    def compute_batch_loss(self, batch):
        raise NotImplementedError

    def quantile_loss(self):
        return torch.tensor(0.0)

    def update(self, force=False):
        pass

    def recompose_model(self, model):
        raise NotImplementedError


class DvcStage1(DvcStage):
    def __init__(self, model, num_pframes, loss_functions):
        super().__init__()
        self.model = model.motion_estimator
        self.num_pframes = num_pframes
        self.loss_functions = loss_functions

    def collect_parameters(self):
        model_parameters = {
            n: p
            for n, p in self.named_parameters()
            if not n.endswith(".quantiles") and p.requires_grad
        }
        return model_parameters, None

    def forward(self, image1, image2):
        return self.model(image1, image2)

    def compute_batch_loss(self, image1, image2):
        flow, outputs = self.model.calculate_flow_with_image_pairs(image1, image2)

        frame_losses = []
        for output in outputs:
            frame_losses.append(self.loss_functions.distortion_fn(output[1], output[0]))

        loss = torch.stack(frame_losses).mean()

        image2_est = ncF.dense_image_warp(image1, flow.permute(0, 2, 3, 1).detach())

        return (
            LossValues(
                distortion_loss=loss, flow_entropy_loss=None, resid_entropy_loss=None
            ),
            OutputTensors(
                flow=flow.detach(),
                image1=image1.detach(),
                image2=image2.detach(),
                image2_est=image2_est.detach(),
            ),
        )

    def recompose_model(self, model):
        model.motion_estimator = self.model
        return model


class DvcStage2(DvcStage):
    def __init__(
        self,
        model,
        num_pframes,
        loss_functions: LossFunctions,
    ):
        super().__init__()
        self.motion_estimator = model.motion_estimator
        self.motion_encoder = model.motion_encoder
        self.motion_entropy_bottleneck = model.motion_entropy_bottleneck
        self.motion_decoder = model.motion_decoder
        self.num_pframes = num_pframes
        if loss_functions.entropy_fn is None:
            raise ValueError("Must specify entropy_fn for compression stage.")
        self.loss_functions = loss_functions

    def forward(self, image1, image2):
        flow = self.motion_estimator(image1, image2)

        latent, sizes = self.motion_encoder(flow)
        latent, probabilities = self.motion_entropy_bottleneck(latent)
        flow = self.motion_decoder(latent, sizes)
        image2_est = ncF.dense_image_warp(image1, flow.permute(0, 2, 3, 1))

        return flow, image2_est, probabilities

    def compute_batch_loss(self, image1, image2):
        assert self.loss_functions.entropy_fn is not None
        assert image1.ndim == image2.ndim == 4

        flow, image2_est, probabilities = self.forward(image1, image2)

        # compute distortion loss
        distortion_loss = self.loss_functions.distortion_fn(image2, image2_est)

        # compute compression loss, average over num pixels
        num_pixels = image1.shape[0] * image1.shape[-2] * image1.shape[-1]
        entropy_loss = self.loss_functions.entropy_fn(probabilities, num_pixels)

        return (
            LossValues(
                distortion_loss=distortion_loss,
                flow_entropy_loss=entropy_loss,
                resid_entropy_loss=None,
            ),
            OutputTensors(
                flow=flow.detach(),
                image1=image1.detach(),
                image2=image2.detach(),
                image2_est=image2_est.detach(),
            ),
        )

    def quantile_loss(self):
        return self.motion_entropy_bottleneck.loss()

    def update(self, force=False):
        return self.motion_entropy_bottleneck.update(force=force)

    def recompose_model(self, model):
        model.motion_estimator = self.motion_estimator
        model.motion_encoder = self.motion_encoder
        model.motion_entropy_bottleneck = self.motion_entropy_bottleneck
        model.motion_decoder = self.motion_decoder
        return model


class DvcStage3(DvcStage):
    def __init__(
        self,
        model,
        num_pframes,
        loss_functions: LossFunctions,
    ):
        super().__init__()
        self.motion_estimator = model.motion_estimator
        self.motion_encoder = model.motion_encoder
        self.motion_entropy_bottleneck = model.motion_entropy_bottleneck
        self.motion_decoder = model.motion_decoder
        self.motion_compensation = model.motion_compensation
        self.num_pframes = num_pframes
        if loss_functions.entropy_fn is None:
            raise ValueError("Must specify entropy_fn for compression stage.")
        self.loss_functions = loss_functions

    def forward(self, image1, image2):
        flow = self.motion_estimator(image1, image2)

        latent, sizes = self.motion_encoder(flow)
        latent, probabilities = self.motion_entropy_bottleneck(latent)
        flow = self.motion_decoder(latent, sizes)
        image2_est = ncF.dense_image_warp(image1, flow.permute(0, 2, 3, 1))

        image2_est = self.motion_compensation(image1, image2_est, flow)

        return flow, image2_est, probabilities

    def compute_batch_loss(self, image1, image2):
        assert self.loss_functions.entropy_fn is not None
        assert image1.ndim == image2.ndim == 4

        flow, image2_est, probabilities = self.forward(image1, image2)

        # compute distortion loss
        distortion_loss = self.loss_functions.distortion_fn(image2, image2_est)

        # compute compression loss, average over num pixels
        num_pixels = image1.shape[0] * image1.shape[-2] * image1.shape[-1]
        entropy_loss = self.loss_functions.entropy_fn(probabilities, num_pixels)

        return (
            LossValues(
                distortion_loss=distortion_loss,
                flow_entropy_loss=entropy_loss,
                resid_entropy_loss=None,
            ),
            OutputTensors(
                flow=flow.detach(),
                image1=image1.detach(),
                image2=image2.detach(),
                image2_est=image2_est.detach(),
            ),
        )

    def quantile_loss(self):
        return self.motion_entropy_bottleneck.loss()

    def update(self, force=False):
        return self.motion_entropy_bottleneck.update(force=force)

    def recompose_model(self, model):
        model.motion_estimator = self.motion_estimator
        model.motion_encoder = self.motion_encoder
        model.motion_entropy_bottleneck = self.motion_entropy_bottleneck
        model.motion_decoder = self.motion_decoder
        model.motion_compensation = self.motion_compensation
        return model


class DvcStage4and5(DvcStage):
    def __init__(self, dvc_model, num_pframes, loss_functions: LossFunctions):
        super().__init__()
        self.model = dvc_model
        self.num_pframes = num_pframes
        if loss_functions.entropy_fn is None:
            raise ValueError("Must specify entropy_fn for compression stage.")
        self.loss_functions = loss_functions

    def forward(self, image1, image2):
        return self.model(image1, image2)

    def compute_batch_loss(self, image1, image2):
        assert self.loss_functions.entropy_fn is not None
        assert image1.ndim == image2.ndim == 4

        output = self.forward(image1, image2)

        # compute distortion loss
        distortion_loss = self.loss_functions.distortion_fn(image2, output.image2_est)

        # compute flow compression loss, average over num pixels
        num_pixels = image1.shape[0] * image1.shape[-2] * image1.shape[-1]
        flow_entropy_loss = self.loss_functions.entropy_fn(
            output.flow_probabilities, num_pixels
        )

        # compute resid compression loss, average over num pixels
        resid_entropy_loss = self.loss_functions.entropy_fn(
            output.resid_probabilities, num_pixels
        )

        return (
            LossValues(
                distortion_loss=distortion_loss,
                flow_entropy_loss=flow_entropy_loss,
                resid_entropy_loss=resid_entropy_loss,
            ),
            OutputTensors(
                flow=output.flow.detach(),
                image1=image1.detach(),
                image2=image2.detach(),
                image2_est=output.image2_est.detach(),
            ),
        )

    def quantile_loss(self):
        return (
            self.model.motion_entropy_bottleneck.loss()
            + self.model.residual_entropy_bottleneck.loss()
        )

    def update(self, force=False) -> bool:
        update1 = self.model.motion_entropy_bottleneck.update(force=force)
        update2 = self.model.residual_entropy_bottleneck.update(force=force)
        return update1 or update2

    def recompose_model(self, model):
        return self.model
