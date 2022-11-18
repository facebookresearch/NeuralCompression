"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Train a diffusion model on images.
Adapted from `improved-diffusion.scripts.image_train`
(https://github.com/openai/improved-diffusion) to support hydra and wandb.

Usage (2 GPUs, Cifar10)
1. Local:
    mpiexec -n 2 python train.py
2. Slurm:
    python train.py -m +mode=submitit hydra.launcher.gpus_per_node=2
"""
import hydra
from bits_back_diffusion.script_util import setup
from improved_diffusion import logger
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig):
    """
    Train a diffusion model.

    Args:
        cfg: Configuration for training.
    """
    model, diffusion, dataloader = setup(cfg, train=True)
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        model=model,
        diffusion=diffusion,
        data=dataloader,
        schedule_sampler={"diffusion": diffusion},
    )

    if cfg.trainer.get("resume_checkpoint"):
        logger.log(
            f"resume training from checkpoint {cfg.trainer.resume_checkpoint}..."
        )
    else:
        logger.log("start training...")
    trainer.run_loop()


if __name__ == "__main__":
    main()
