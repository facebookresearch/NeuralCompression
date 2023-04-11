# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional, Tuple, Union

import hydra
import pytorch_lightning
import torch
import torch.nn.functional as F
from datamodules.video_data_api import VideoData
from model_pipeline import VCTPipeline
from omegaconf import DictConfig
from torch import Tensor
from torchmetrics.functional import peak_signal_noise_ratio
from utils.hydra_tools import OmegaConf  # ! Needed for resolvets to take effect


class VCTModule(pytorch_lightning.LightningModule):
    """Encapsulates model + loss + optimizer + metrics in a PL module"""

    def __init__(self, model: VCTPipeline, cfg: DictConfig) -> None:
        super().__init__()
        self.model = model
        self.cfg_optim = getattr(cfg, "optim", None)
        self.cfg_scheduler = getattr(cfg, "scheduler", None)
        self.lr_annealing_frequency = getattr(
            getattr(cfg, "training_loop", None), "lr_annealing_frequency", None
        )
        self._distortion_loss_scaling = getattr(
            getattr(cfg, "training_loop", None), "distortion_lambda", None
        )
        if self._distortion_loss_scaling is not None:
            self._distortion_loss_scaling = self._distortion_loss_scaling * (255**2)

    def forward(self, batch: VideoData) -> Tensor:
        return self.model(batch)

    def rate_loss(self, video: VideoData, rate_args, per_frame: bool = True) -> Tensor:
        # average over batch dim but not over frame within the video
        B, T, _, H, W = video.shape
        num_pixels = B * H * W if per_frame else B * H * W * T
        total_bits = self.model.compute_rate(*rate_args, per_frame=per_frame)
        return total_bits / num_pixels

    def training_step(
        self, batch, batch_idx: int, optimizer_idx: Optional[int] = None
    ) -> Tensor:
        recon_frames, rate_args = self(batch)  # [B, T, 3, H, W], [Tensor]

        dist_loss_frame = F.mse_loss(
            recon_frames, batch.video_tensor, reduction="none"
        ).mean(dim=(0, 2, 3, 4))
        rate_loss_frame = self.rate_loss(batch, rate_args, per_frame=True)  # [T]
        weights = 10.0 * torch.ones_like(rate_loss_frame)
        weights[0] = 1.0
        combined_loss_frame = (
            self._distortion_loss_scaling * dist_loss_frame + rate_loss_frame
        )
        combined_loss_weighted = (combined_loss_frame * weights).mean()

        with torch.no_grad():
            psnr = peak_signal_noise_ratio(
                recon_frames, batch.video_tensor, data_range=1.0, dim=(2, 3, 4)
            ).item()
            psnr_int = peak_signal_noise_ratio(
                (recon_frames * 255 + 0.5).to(torch.uint8),
                (batch.video_tensor * 255 + 0.5).to(torch.uint8),
                data_range=255,
                dim=(2, 3, 4),
            ).item()
        self.log_dict(
            {
                "distortion_loss": dist_loss_frame.mean(),
                "rate_loss": rate_loss_frame.mean(),
                "combined_loss": combined_loss_frame.mean(),
                "combined_loss_weighted": combined_loss_weighted,
                "PSNR": psnr,
                "PSNR_int": psnr_int,
            },
            sync_dist=True,
        )
        return combined_loss_weighted

    def validation_step(self, batch, batch_idx) -> None:
        recon_frames, rate_args = self(batch)

        dist_loss_frame = F.mse_loss(
            recon_frames, batch.video_tensor, reduction="none"
        ).mean(dim=(0, 2, 3, 4))
        rate_loss_frame = self.rate_loss(batch, rate_args, per_frame=True)  # [T]
        weights = 10.0 * torch.ones_like(rate_loss_frame)
        weights[0] = 1.0
        combined_loss_frame = (
            self._distortion_loss_scaling * dist_loss_frame + rate_loss_frame
        )
        combined_loss_weighted = (combined_loss_frame * weights).mean()

        self.log_dict(
            {
                "val_distortion_loss": dist_loss_frame.mean().item(),
                "val_rate_loss": rate_loss_frame.mean().item(),
                "val_combined_loss": combined_loss_frame.mean().item(),
                "val_combined_loss_weighted": combined_loss_weighted,
                "val_PSNR": peak_signal_noise_ratio(
                    recon_frames, batch.video_tensor, data_range=1.0, dim=(2, 3, 4)
                ).item(),
                "val_PSNR_int": peak_signal_noise_ratio(
                    (recon_frames * 255 + 0.5).to(torch.uint8),
                    (batch.video_tensor * 255 + 0.5).to(torch.uint8),
                    data_range=255,
                    dim=(2, 3, 4),
                ).item(),
            },
            sync_dist=True,
        )

    def test_step(
        self,
        batch: VideoData,
        batch_idx: int,
        run_fwd: bool = True,
    ):
        B, T, C, H, W = batch.shape
        assert B == 1, "Metrics calculation only supported for batch size 1."
        assert self.training is False

        with torch.no_grad():
            video_ref_uint8 = (batch.video_tensor[0] * 255 + 0.5).to(torch.uint8)
            if run_fwd:
                self.model._code_to_strings = False
                forward_pass = self.model(batch)
                recon, rate_args = forward_pass[-2], forward_pass[-1][0]

                # take the single batch and clamp it:
                recon_clamped = torch.clamp(recon[0], 0.0, 1.0)
                # caveat: real compression people use YCbCr
                recon_uint8 = (recon_clamped * 255 + 0.5).to(torch.uint8)
                cts_psnr = peak_signal_noise_ratio(
                    recon_uint8, video_ref_uint8, data_range=255
                ).item()
                cts_rate = self.model.compute_rate(rate_args).item() / (H * W * T)
                # Compute the rate excluding the first frame since we are technically
                # not compressing it. Generally even if the compression is not great
                # for that first frame, its coding cost will amportize over the video
                # sequence length
                cts_rate_x1 = self.model.compute_rate(rate_args[1:]).item() / (
                    H * W * (T - 1)
                )
                self.log_dict(
                    {
                        f"cts_psnr": cts_psnr,
                        f"cts_rate": cts_rate,
                        f"cts_rate_x1": cts_rate_x1,
                    },
                    sync_dist=True,
                )

            *output, bits_per_frame = self.model.compress_video(batch, force_cpu=False)
            recon = self.model.decompress_video(
                batch.shape, bottleneck_args=output, force_cpu=False
            )
            # clamp the reconstructed video (it's batch size 1, so pick [0])
            recon_clamped = torch.clamp(recon[0], 0.0, 1.0)

            # PSNR
            # caveat: real compression people use YCbCr
            recon_uint8 = (recon_clamped * 255 + 0.5).to(torch.uint8)
            psnr = peak_signal_noise_ratio(
                recon_uint8, video_ref_uint8, data_range=255
            ).item()
            # bits per pixel
            bpp = sum(bits_per_frame) / (H * W * T)
            # Compute the rate excluding the first frame since we are technically
            # not compressing it. Generally even if the compression is not great
            # for that first frame, its coding cost will amportize over the video
            # sequence length
            bpp_xfirst = sum(bits_per_frame[1:]) / (H * W * (T - 1))
            print(psnr, bpp, bpp_xfirst)
            self.log_dict(
                {f"psnt_int": psnr, f"bpp": bpp, f"bpp_xfirst": bpp_xfirst},
                sync_dist=True,
            )
        return psnr, bpp, bpp_xfirst

    def configure_optimizers(
        self,
    ) -> Union[
        torch.optim.Optimizer,  # Single optimizer
        Tuple[torch.optim.Optimizer, torch.optim.Optimizer],  # Tuple or list of optim
        List[torch.optim.Optimizer],
        dict,  # "optimizer" key, and (optionally) an "lr_scheduler"
        Any,  # 2 lists: first with optimizers, second has LR schedulers; or Tuple[Dict]
    ]:
        # PL allows return type to be tuple/list/dict/two lists/tuple of dicts/None
        model_params = (
            p
            for n, p in self.named_parameters()
            if not n.endswith(".quantiles") and p.requires_grad
        )

        base_optim = hydra.utils.instantiate(self.cfg_optim, params=model_params)
        base_scheduler = hydra.utils.instantiate(
            self.cfg_scheduler, optimizer=base_optim
        )
        scheduler = {
            "scheduler": base_scheduler,
            "interval": "step",
            "frequency": self.lr_annealing_frequency,
        }
        return {"optimizer": base_optim, "lr_scheduler": scheduler}
