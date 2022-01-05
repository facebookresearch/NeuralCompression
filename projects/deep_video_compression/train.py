# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import _optical_flow_to_color
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from data_module import Vimeo90kSeptupletLightning
from dvc_module import DvcModule
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from neuralcompression.models import DVC


class WandbImageCallback(pl.Callback):
    """
    Logs the input and output images of a module.

    Images are stacked into a mosaic, with different clips from top to bottom
    and time progressing from left to right.

    Args:
        batch: A set of images to log at the end of each validation epoch.
    """

    def __init__(self, batch):
        super().__init__()
        self.batch = batch

    def append_images(self, image_dict, image_group):
        keys = ("flow", "image1", "image2", "image2_est")
        for i, key in enumerate(keys):
            if image_group[i] is not None:
                image_dict.setdefault(key, []).append(image_group[i])
        return image_dict

    def log_images(self, trainer, base_key, image_dict, global_step):
        # check if we have images to log
        # if we do, then concatenate time along x-axis and batch along y-axis
        # and write
        keys = ("flow", "image1", "image2", "image2_est")
        for key in keys:
            if image_dict.get(key) is not None:
                caption = f"{key} (y-axis: batch, x-axis: time)"
                mosaic = torch.cat(image_dict[key], dim=-1)
                mosaic = torch.cat(list(mosaic), dim=-2)
                if key == "flow":
                    mosaic = _optical_flow_to_color.optical_flow_to_color(
                        mosaic.unsqueeze(0)
                    )[0]
                mosaic = torch.clip(mosaic, min=0, max=1.0)
                trainer.logger.experiment.log(
                    {
                        f"{base_key}/{key}": wandb.Image(mosaic, caption=caption),
                        "global_step": global_step,
                    }
                )

    def on_validation_end(self, trainer, pl_module):
        image_dict = {}
        batch = self.batch.to(device=pl_module.device, dtype=pl_module.dtype)
        batch, _ = pl_module.compress_iframe(batch)  # bpp_total w/o grads
        image1 = batch[:, 0]
        for i in range(pl_module.num_pframes):
            image2 = batch[:, i + 1]
            _, images = pl_module.model.compute_batch_loss(image1, image2)
            image1 = images.image2_est  # images are detached

            image_dict = self.append_images(image_dict, images)

        self.log_images(
            trainer,
            f"log_images_stage_{pl_module.training_stage}",
            image_dict,
            pl_module.global_step,
        )


def merge_configs(cfg1, cfg2):
    """Handy config merger based on dictionaries."""
    new_cfg = cfg1.copy()
    OmegaConf.set_struct(new_cfg, False)
    new_cfg.update(cfg2)

    return new_cfg


def run_training_stage(stage, root, model, data, logger, image_logger, cfg):
    """Run a single training stage based on the stage config."""
    print(f"training stage: {stage}")
    stage_cfg = cfg.training_stages[stage]
    if stage_cfg.save_dir is None:
        save_dir = root / stage
    else:
        save_dir = Path(stage_cfg.save_dir)

    if (
        not cfg.checkpoint.overwrite
        and not cfg.checkpoint.resume_training
        and len(list(save_dir.glob("*.ckpt"))) > 0
    ):
        raise RuntimeError(
            "Checkpoints detected in save directory: set resume_training=True "
            "to restore trainer state from these checkpoints, "
            "or set overwrite=True to ignore them."
        )

    save_dir.mkdir(exist_ok=True, parents=True)
    last_checkpoint = save_dir / "last.ckpt"
    if not last_checkpoint.exists() or cfg.checkpoint.overwrite is True:
        last_checkpoint = None

    lightning_model = DvcModule(model, **merge_configs(cfg.module, stage_cfg.module))

    trainer = pl.Trainer(
        **merge_configs(cfg.trainer, stage_cfg.trainer),
        logger=logger,
        callbacks=[
            LearningRateMonitor(),
            ModelCheckpoint(dirpath=save_dir, **cfg.checkpoint.model_checkpoint),
            image_logger,
        ],
        resume_from_checkpoint=last_checkpoint,
    )

    trainer.fit(lightning_model, datamodule=data)

    return lightning_model.recompose_model(model)


@hydra.main(config_path="config", config_name="base")
def main(cfg: DictConfig):
    root = Path(cfg.logging.save_root)  # if relative, uses Hydra outputs dir
    model = DVC(**cfg.model)
    logger = WandbLogger(
        save_dir=str(root.absolute()),
        project="DVC",
        config=OmegaConf.to_container(cfg),  # saves the Hydra config to wandb
    )
    data = Vimeo90kSeptupletLightning(
        frames_per_group=7,
        **cfg.data,
        pin_memory=cfg.ngpu != 0,
    )

    # set up image logging
    rng = np.random.default_rng(cfg.logging.image_seed)
    data.setup()
    val_dataset = data.val_dataset
    log_image_indices = rng.permutation(len(val_dataset))[: cfg.logging.num_log_images]
    log_images = torch.stack([val_dataset[ind] for ind in log_image_indices])
    image_logger = WandbImageCallback(log_images)

    # run through each stage and optimize
    for stage in sorted(cfg.training_stages.keys()):
        model = run_training_stage(stage, root, model, data, logger, image_logger, cfg)


if __name__ == "__main__":
    main()
