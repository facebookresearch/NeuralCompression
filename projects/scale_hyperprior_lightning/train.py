# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from scale_hyperprior import ScaleHyperpriorLightning
from vimeo import Vimeo90kSeptupletLightning

from neuralcompression.models import ScaleHyperprior


@hydra.main(config_path="config", config_name="base")
def main(cfg: DictConfig):

    save_dir: Path = Path(hydra.utils.get_original_cwd()) / cfg.save_dir

    if (
        not cfg.overwrite
        and not cfg.resume_training
        and len(list(save_dir.glob("*.ckpt"))) > 0
    ):
        raise RuntimeError(
            "Checkpoints detected in save directory: set resume_training=True"
            " to restore trainer state from these checkpoints, or set overwrite=True"
            " to ignore them."
        )

    save_dir.mkdir(exist_ok=True, parents=True)
    last_checkpoint = save_dir / "last.ckpt"

    model = ScaleHyperprior(**cfg.model)
    lightning_model = ScaleHyperpriorLightning(model, **cfg.training_loop)

    data = Vimeo90kSeptupletLightning(**cfg.data, pin_memory=cfg.ngpu != 0)

    loggers = [hydra.utils.instantiate(logger_cfg) for logger_cfg in cfg.loggers]
    trainer = Trainer(
        **cfg.trainer,
        logger=loggers,
        callbacks=[
            LearningRateMonitor(),
            ModelCheckpoint(**cfg.save_model),
        ],
        resume_from_checkpoint=last_checkpoint
        if last_checkpoint.exists() and cfg.resume_training
        else None,
    )

    trainer.fit(lightning_model, datamodule=data)


if __name__ == "__main__":
    main()
