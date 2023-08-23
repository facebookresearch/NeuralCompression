# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import logging
import math
import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Union
from warnings import warn

import hydra
import lightning.pytorch as pl
import torch
import torch.nn as nn
import yaml
from gan_compression_module import GANCompressionModule
from image_module import ImageModule
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import DictConfig, OmegaConf
from open_images_datamodule import OpenImagesDataModule
from target_compression_module import TargetRateCompressionModule
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from neuralcompression.loss_fn import TargetRateConfig
from neuralcompression.models import HyperpriorAutoencoderBase

LOGGER = logging.getLogger(__file__)

try:
    import wandb
    from lightning.pytorch.loggers import WandbLogger

    HAS_WANDB = True
except ModuleNotFoundError:
    HAS_WANDB = False
    LOGGER.warn("wandb not found - disabling image logging.")


class WandbImageCallback(pl.Callback):
    """
    Logs the input and output images of a module.
    Images are stacked into a mosaic, with different clips from top to bottom
    and time progressing from left to right.

    Args:
        train_batch: A set of images to log at the end of each training epoch.
        eval_batch: A set of images to log at the end of each validation epoch.
    """

    def __init__(self, train_batch: Tensor, eval_batch: Tensor):
        super().__init__()
        self.train_batch = train_batch
        self.eval_batch = eval_batch
        self.key_set = ("image", "image_compressed")

    def log_images(self, trainer, base_key, image_dict, global_step):
        # check if we have images to log
        # if we do, then concatenate time along x-axis and batch along y-axis
        # and write
        for key in self.key_set:
            if image_dict.get(key) is not None:
                caption = f"{key}"
                trainer.logger.experiment.log(
                    {
                        f"{base_key}/{key}": wandb.Image(
                            image_dict[key], caption=caption
                        ),
                        "global_step": global_step,
                    }
                )

    def _compute_input_output_batch(self, pl_module: pl.LightningModule, batch: Tensor):
        batch = batch.to(device=pl_module.device, dtype=pl_module.dtype)
        with torch.no_grad():
            outputs = pl_module(batch).image

        return {
            "image": make_grid(batch),
            "image_compressed": make_grid(torch.clip(outputs, min=0, max=1.0)),
        }

    def on_validation_end(self, trainer, pl_module):
        self.log_images(
            trainer,
            "train_images",
            self._compute_input_output_batch(pl_module, self.train_batch),
            trainer.global_step,
        )
        self.log_images(
            trainer,
            "eval_images",
            self._compute_input_output_batch(pl_module, self.eval_batch),
            trainer.global_step,
        )

    def on_train_epoch_end(self, trainer, pl_module):
        self.log_images(
            trainer,
            "model.training_train_images",
            self._compute_input_output_batch(pl_module, self.train_batch),
            trainer.global_step,
        )
        self.log_images(
            trainer,
            "model.training_eval_images",
            self._compute_input_output_batch(pl_module, self.eval_batch),
            trainer.global_step,
        )


def pretrained_state_dict(checkpoint_file: str, prefix: str = "model."):
    state_dict = torch.load(checkpoint_file, map_location=torch.device("cpu"))[
        "state_dict"
    ]
    state_dict = {
        k[len(prefix) :]: v for k, v in state_dict.items() if k[: len(prefix)] == prefix
    }

    return state_dict


def configure_and_load_state_dict(model: nn.Module, cfg: DictConfig):
    if cfg.get("torchhub") is not None:
        LOGGER.info("Loading model from torchhub")

        model = torch.hub.load(
            cfg.torchhub.github, cfg.torchhub.model, force_reload=True
        )
    else:
        LOGGER.info("Loading pretrained projector from " f"{cfg.path}")
        state_dict = pretrained_state_dict(cfg.path)
        model.load_state_dict(state_dict)

    return model


def build_model(
    cfg: DictConfig,
) -> Tuple[HyperpriorAutoencoderBase, Optional[TargetRateConfig]]:
    model: HyperpriorAutoencoderBase = hydra.utils.instantiate(cfg.model)
    if cfg.get("pretrained_autoencoder") is not None:
        model = configure_and_load_state_dict(model, cfg.pretrained_autoencoder)

    return model


def build_module(cfg: DictConfig) -> ImageModule:
    """Builds the pl training module."""
    model = build_model(cfg)

    if cfg.distortion_loss.get("_target_") is not None:
        distortion_fn = hydra.utils.instantiate(cfg.distortion_loss)
    else:
        raise ValueError("Misconfigured loss function.")

    # for HiFiC-style rate targeting
    target_rate_config = hydra.utils.instantiate(cfg.rate_target)

    if cfg.get("discriminator") is None:
        module = TargetRateCompressionModule(
            model=model,
            target_rate_config=target_rate_config,
            optimizer_config=cfg.optimizer,
            distortion_loss=distortion_fn,
            **cfg.lightning_module,
        )
    else:
        discriminator = hydra.utils.instantiate(cfg.discriminator.model)
        discriminator_loss = hydra.utils.instantiate(
            cfg.discriminator.discriminator_loss
        )
        generator_loss = hydra.utils.instantiate(cfg.discriminator.generator_loss)
        if cfg.get("latent_projector") is not None:
            latent_autoencoder: nn.Module = hydra.utils.instantiate(
                cfg.latent_projector.autoencoder
            )

            if cfg.get("pretrained_latent_autoencoder") is None:
                warn(
                    "No pretrained_latenet_projector specified, but latent_projector "
                    "is specified! This will likely give bad results."
                )
            else:
                latent_autoencoder = configure_and_load_state_dict(
                    latent_autoencoder, cfg.pretrained_latent_autoencoder
                )

            latent_projector = hydra.utils.instantiate(
                cfg.latent_projector.projector, latent_autoencoder
            )
            assert latent_projector is not None
            latent_projector = latent_projector.eval()
            for param in latent_projector.parameters():
                param.requires_grad_(False)
        else:
            latent_projector = None

        module = GANCompressionModule(
            model=model,
            target_rate_config=target_rate_config,
            discriminator=discriminator,
            discriminator_loss=discriminator_loss,
            generator_loss=generator_loss,
            distortion_loss=distortion_fn,
            latent_projector=latent_projector,
            optimizer_config=cfg.optimizer,
            **cfg.lightning_module,
        )

    return module


def build_wandb_image_logger(cfg: DictConfig, data_module: OpenImagesDataModule):
    """Construct logger for sending images to wandb."""
    log_cache_dir = Path(cfg.image_logs.log_cache_dir)
    if not log_cache_dir.exists():
        log_cache_dir.mkdir(parents=True)

    sha = hashlib.sha256()
    sha.update(str(cfg.data._target_).encode())
    sha.update(str(cfg.image_logs.train_log_indices).encode())
    sha.update(str(cfg.image_logs.val_log_indices).encode())
    pkl_file = log_cache_dir / (sha.hexdigest() + ".pkl")
    if pkl_file.exists():
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)

        train_log_images = data["train_log_images"]
        eval_log_images = data["eval_log_images"]
    else:
        data_module.setup()
        train_log_images = torch.stack(
            [data_module.train_dataset[ind] for ind in cfg.image_logs.train_log_indices]
        )
        eval_log_images = torch.stack(
            [data_module.eval_dataset[ind] for ind in cfg.image_logs.val_log_indices]
        )
        with open(pkl_file, "wb") as f:
            pickle.dump(
                {
                    "train_log_images": train_log_images,
                    "eval_log_images": eval_log_images,
                },
                f,
            )

    return WandbImageCallback(train_log_images, eval_log_images)


def fetch_checkpoint(cfg: DictConfig) -> Optional[str]:
    """Look in the checkpoint directory and fetch any checkpoints."""
    checkpoint_dir = Path(cfg.checkpoint.callback.dirpath).absolute()
    if (
        not cfg.checkpoint.overwrite
        and not cfg.checkpoint.resume_training
        and len(list(checkpoint_dir.glob("*.ckpt"))) > 0
    ):
        raise RuntimeError(
            "Checkpoints detected in save directory: set resume_training=True "
            "to restore trainer state from these checkpoints, "
            "or set overwrite=True to ignore them."
        )

    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    last_checkpoint: Optional[Path] = None
    if (checkpoint_dir / "last.ckpt").exists() and cfg.checkpoint.overwrite is False:
        last_checkpoint = checkpoint_dir / "last.ckpt"

    return (
        str(last_checkpoint.absolute()) if isinstance(last_checkpoint, Path) else None
    )


def build_trainer(
    cfg: DictConfig, data_module: OpenImagesDataModule, module: ImageModule
) -> pl.Trainer:
    """Construct the pl trainer."""
    log_dir = Path.cwd().absolute() / "wandb_logs"
    log_dir.mkdir(exist_ok=True, parents=True)
    if cfg.logger.group == "dev":
        name = "/".join([Path.cwd().parent.name, Path.cwd().name])
    else:
        name = Path.cwd().name

    if HAS_WANDB:
        # create an id for this run based on hash of config string
        sha = hashlib.sha256()
        sha.update(str(Path.cwd()).encode())
        wandb_id = sha.hexdigest()

        logger = WandbLogger(
            name=name,
            save_dir=str(log_dir),
            project="illm",
            id=wandb_id,
            config=OmegaConf.to_container(cfg),
            **cfg.logger,
        )

        image_logger = build_wandb_image_logger(cfg, data_module)
    else:
        logger = None
        image_logger = pl.Callback()

    # sometimes we have unused parameters
    strategy: Union[DDPStrategy, str]
    if cfg.training_mode == "train":
        strategy = DDPStrategy(find_unused_parameters=True)
    else:
        strategy = "auto"

    # adjust number of steps based on how many optimizers we have
    num_opts = len(module.configure_optimizers()[0])
    trainer_cfg = {**cfg.trainer}
    orig_steps = trainer_cfg["max_steps"]
    new_steps = math.ceil(trainer_cfg["max_steps"] * num_opts)
    LOGGER.info(
        f"found {num_opts} optimizers, changing max steps "
        f"from {orig_steps} to {new_steps}"
    )
    trainer_cfg["max_steps"] = new_steps

    trainer = pl.Trainer(
        **trainer_cfg,
        logger=logger,
        strategy=strategy,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(**cfg.checkpoint.callback),
            image_logger,
        ],
    )

    return trainer


def build_test_dataloaders(cfg: DictConfig) -> Tuple[List[str], List[DataLoader]]:
    """Loop over and construct any test dataloaders."""
    dataset_names: List[str] = []
    dataloaders: List[DataLoader] = []
    for _, v in cfg.test_dataloaders.items():
        dataset_names.append(v.name)
        dataloaders.append(hydra.utils.instantiate(v.class_params))

    return dataset_names, dataloaders


@hydra.main(config_path="conf/", config_name="config.yaml", version_base="1.1")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)
    data_module = hydra.utils.instantiate(cfg.data, val_batch_size=None)
    lightning_module = build_module(cfg)

    last_checkpoint = fetch_checkpoint(cfg)
    trainer = build_trainer(cfg, data_module, lightning_module)

    if cfg.training_mode == "train":
        trainer.fit(lightning_module, ckpt_path=last_checkpoint, datamodule=data_module)
    else:
        assert last_checkpoint is not None
        LOGGER.info(f"loading from {last_checkpoint}")
        lightning_module.model.load_state_dict(pretrained_state_dict(last_checkpoint))
        lightning_module.eval()
        if cfg.ngpu > 1:
            raise ValueError("Number of GPUs must be <=1 for test")
        dataset_names, test_dataloaders = build_test_dataloaders(cfg)
        for dataset_name, test_dataloader in zip(dataset_names, test_dataloaders):
            lightning_module.update_test_dataset_name(dataset_name)
            LOGGER.info(f"Starting tests on {dataset_name}")
            trainer.test(
                model=lightning_module,
                dataloaders=[test_dataloader],
            )


if __name__ == "__main__":
    main()
