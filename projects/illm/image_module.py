# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional, Union

import torch
from lightning.pytorch import LightningModule
from torch import Tensor
from torchmetrics import MeanSquaredError, PeakSignalNoiseRatio
from torchmetrics.image import (
    FrechetInceptionDistance,
    LearnedPerceptualImagePatchSimilarity,
)

import neuralcompression.functional as ncF
import neuralcompression.metrics as ncm
from neuralcompression import HyperpriorOutput
from neuralcompression.metrics import MultiscaleStructuralSimilarity


class ImageModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.custom_print_logger = logging.getLogger(self.__class__.__name__)

        # train metrics
        self.train_psnr = PeakSignalNoiseRatio(data_range=255.0)
        self.train_mse = MeanSquaredError()
        self.train_msssim = MultiscaleStructuralSimilarity(data_range=255.0)

        # validation metrics
        self.eval_psnr = PeakSignalNoiseRatio(data_range=255.0)
        self.eval_mse = MeanSquaredError()
        self.eval_msssim = MultiscaleStructuralSimilarity(data_range=255.0)
        self.eval_lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.eval_fid = FrechetInceptionDistance()

        # test metrics, set these to None to start
        self.test_metrics_set = False
        self.test_mse: Optional[MeanSquaredError] = None
        self.test_msssim: Optional[MultiscaleStructuralSimilarity] = None
        self.test_lpips: Optional[LearnedPerceptualImagePatchSimilarity] = None
        self.test_dists: Optional[ncm.DeepImageStructureTextureSimilarity] = None
        self.test_fid: Optional[ncm.FrechetInceptionDistance] = None
        self.test_fid_swav: Optional[ncm.FrechetInceptionDistanceSwAV] = None
        self.test_kid: Optional[ncm.KernelInceptionDistance] = None

        self.hific_mse_param = 0.075 * 2**-5
        self.hific_lpips_param = 1.0
        self.test_dataset_name: Optional[str] = None
        self.patch_size = 256

    def set_test_metrics(self):
        self.test_mse = MeanSquaredError().to(self.device)
        self.test_msssim = MultiscaleStructuralSimilarity(data_range=255.0).to(
            self.device
        )
        self.test_lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(
            self.device
        )
        self.test_dists = ncm.DeepImageStructureTextureSimilarity().to(self.device)
        self.test_fid = ncm.FrechetInceptionDistance().to(self.device)
        self.test_fid_swav = ncm.FrechetInceptionDistanceSwAV().to(self.device)
        self.test_kid = ncm.KernelInceptionDistance().to(self.device)

    def update_test_dataset_name(self, test_dataset_name: str):
        self.test_dataset_name = test_dataset_name
        if not self.test_metrics_set:
            self.set_test_metrics()
            self.test_metrics_set = True

    def reset_all_metrics(self):
        self.train_psnr.reset()
        self.train_mse.reset()
        self.train_msssim.reset()
        self.eval_psnr.reset()
        self.eval_mse.reset()
        self.eval_msssim.reset()
        self.eval_lpips.reset()
        self.eval_fid.reset()
        if self.test_metrics_set:
            self.test_mse.reset()
            self.test_msssim.reset()
            self.test_lpips.reset()
            self.test_dists.reset()
            self.test_fid.reset()
            self.test_fid_swav.reset()
            self.test_kid.reset()

    def _update_test_fid_kid(
        self, input_images: Tensor, pred: Tensor, clic_eval: bool = False
    ):
        assert self.test_kid is not None
        assert self.test_fid is not None
        assert self.test_fid_swav is not None
        if clic_eval:
            # this applies the FID/KID calculations from Mentzer 2020
            ncm.update_patch_fid(
                input_images=input_images,
                pred=pred,
                fid_metric=self.test_fid,
                fid_swav_metric=self.test_fid_swav,
                kid_metric=self.test_kid,
                patch_size=self.patch_size,
            )
        else:
            self.test_fid.update(
                ncF.image_to_255_scale(input_images, dtype=torch.uint8), real=True
            )
            self.test_fid.update(
                ncF.image_to_255_scale(pred, dtype=torch.uint8), real=False
            )
            self.test_fid_swav.update(
                ncF.image_to_255_scale(input_images, dtype=torch.uint8), real=True
            )
            self.test_fid_swav.update(
                ncF.image_to_255_scale(pred, dtype=torch.uint8), real=False
            )
            self.test_kid.update(
                ncF.image_to_255_scale(input_images, dtype=torch.uint8), real=True
            )
            self.test_kid.update(
                ncF.image_to_255_scale(pred, dtype=torch.uint8), real=False
            )

    def log_quality_metrics(
        self,
        output: Union[HyperpriorOutput, Tensor],
        input_images: Tensor,
        prefix,
        dataset_name: Optional[str] = None,
    ):
        # update metrics
        if isinstance(output, Tensor):
            pred = output.clamp(0.0, 1.0)
        else:
            assert output.image is not None
            pred = output.image.clamp(0.0, 1.0)
        rescaled_pred = ncF.image_to_255_scale(pred)
        rescaled_input_images = ncF.image_to_255_scale(input_images)

        batch_size = pred.shape[0]

        if prefix == "train":
            with torch.no_grad():
                self.train_psnr(rescaled_pred, rescaled_input_images)
                self.train_msssim(rescaled_pred, rescaled_input_images)
                self.train_mse(
                    rescaled_pred.view(batch_size, -1),
                    rescaled_input_images.view(batch_size, -1),
                )
            self.log(f"{prefix}/psnr", self.train_psnr, prog_bar=True)
            self.log(f"{prefix}/mse", self.train_mse, prog_bar=True)
            self.log(f"{prefix}/ms_ssim", self.train_msssim)
        elif prefix == "val":
            self.eval_psnr(rescaled_pred, rescaled_input_images)
            self.eval_msssim(rescaled_pred, rescaled_input_images)
            mse_val = self.eval_mse(
                rescaled_pred.view(batch_size, -1),
                rescaled_input_images.view(batch_size, -1),
            )
            lpips_val = self.eval_lpips(pred, input_images)
            hific_distortion = (
                self.hific_mse_param * mse_val + self.hific_lpips_param * lpips_val
            )
            self.eval_fid.update(
                ncF.image_to_255_scale(input_images, dtype=torch.uint8), real=True
            )
            self.eval_fid.update(
                ncF.image_to_255_scale(pred, dtype=torch.uint8), real=False
            )
            self.log_dict(
                {
                    f"{prefix}/psnr": self.eval_psnr,
                    f"{prefix}/mse": self.eval_mse,
                    f"{prefix}/ms_ssim": self.eval_msssim,
                    f"{prefix}/lpips": self.eval_lpips,
                    f"{prefix}/hific_distortion": hific_distortion,
                },
                sync_dist=True,
            )
        elif prefix == "test":
            assert self.test_msssim is not None
            assert self.test_mse is not None
            assert self.test_lpips is not None
            assert self.test_dists is not None
            if dataset_name is None:
                raise ValueError("Must pass dataset_name for test")

            psnr_val = ncm.calc_psnr(rescaled_pred, rescaled_input_images)
            self.test_msssim(rescaled_pred, rescaled_input_images)
            mse_val = self.test_mse(
                rescaled_pred.view(batch_size, -1),
                rescaled_input_images.view(batch_size, -1),
            )
            lpips_val = self.test_lpips(pred, input_images)
            self.test_dists(pred, input_images)
            hific_distortion = (
                self.hific_mse_param * mse_val + self.hific_lpips_param * lpips_val
            )
            clic_eval = "clic" in dataset_name or "div2k" in dataset_name
            self._update_test_fid_kid(
                input_images=input_images, pred=pred, clic_eval=clic_eval
            )
            self.log_dict(
                {
                    f"{prefix}/{dataset_name}_psnr": psnr_val,
                    f"{prefix}/{dataset_name}_mse": self.test_mse,
                    f"{prefix}/{dataset_name}_ms_ssim": self.test_msssim,
                    f"{prefix}/{dataset_name}_lpips": self.test_lpips,
                    f"{prefix}/{dataset_name}_dists": self.test_dists,
                    f"{prefix}/{dataset_name}_hific_distortion": hific_distortion,
                }
            )
        else:
            raise ValueError(f"Unrecognized prefix {prefix}")

    def on_validation_epoch_start(self) -> None:
        self.custom_print_logger.info("Resetting metrics on validation epoch start")
        self.reset_all_metrics()

    def on_test_epoch_start(self) -> None:
        self.custom_print_logger.info("Resetting metrics on test epoch start")
        self.reset_all_metrics()

    def on_validation_epoch_end(self) -> None:
        # sometimes this fails on job launch
        try:
            self.log("val/fid", self.eval_fid.compute(), sync_dist=True)
            self.eval_fid.reset()
        except IndexError:
            self.eval_fid.reset()
        except ValueError:
            self.eval_fid.reset()
        except Exception:
            raise

    def on_test_epoch_end(self) -> None:
        dataset_name = self.test_dataset_name
        assert self.test_fid is not None
        assert self.test_fid_swav is not None
        assert self.test_kid is not None
        if dataset_name == "kodak":
            self.log(f"test/{dataset_name}_fid", 0.0)
            self.log(f"test/{dataset_name}_fid_swav", 0.0)
            self.log(f"test/{dataset_name}_kid", 0.0)
        else:
            self.log(f"test/{dataset_name}_fid", self.test_fid.compute())
            self.log(f"test/{dataset_name}_fid_swav", self.test_fid_swav.compute())
            self.log(f"test/{dataset_name}_kid", self.test_kid.compute()[0])
