"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Encode images using Bits-Back coding.

Usage (1 GPU, pretrained model, Cifar10):
1. Local:
    python encode.py
2. Slurm:
    python encode.py -m +mode=submitit

When using multiprocessing, each process independently
compresses a subset of the data.
"""
from itertools import islice
from pathlib import Path
from typing import Dict, Iterator, List, Tuple, Union

import hydra
import numpy as np
import torch
import torch.distributed as dist
import wandb
from bits_back_diffusion.codec import BitsBackCodec
from bits_back_diffusion.script_util import plot_evolution, setup, update_checkpoint
from improved_diffusion import logger
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="config", config_name="encode")
def main(cfg: DictConfig):
    """
    Encode data using a diffusion model.

    Args:
        cfg: Configuration for encoding.
    """
    model, diffusion, dataloader = setup(cfg, log_ranks=True)

    codec = hydra.utils.instantiate(
        cfg.codec,
        model={
            "model": model,
            "diffusion": diffusion,
            "clip_denoised": cfg.evaluator.clip_denoised,
        },
    )

    # load checkpoint
    update_checkpoint(cfg, "encode_checkpoint", prefix="encode-ckpt", suffix="npz")
    if cfg.get("encode_checkpoint"):
        logger.log(f"load encode checkpoint {cfg.encode_checkpoint}...")
        codec.load_state_dict(np.load(cfg.encode_checkpoint))

    if (
        codec.data_shape[0] * codec.data_count * dist.get_world_size()
        < cfg.evaluator.num_samples
    ):
        logger.log(f"encode with model checkpoint {cfg.evaluator.model_path}...")
        encode(
            codec,
            islice(dataloader, codec.data_count, None),
            cfg.evaluator.num_samples,
            cfg.save_interval,
        )

    if cfg.check:
        update_checkpoint(cfg, "decode_checkpoint", prefix="decode-ckpt", suffix="npz")
        if cfg.get("decode_checkpoint"):
            logger.log(f"load decode checkpoint {cfg.decode_checkpoint}...")
            codec.load_state_dict(np.load(cfg.decode_checkpoint))

        save_dir = Path(logger.get_dir())
        logger.log(f"decode with model checkpoint {cfg.evaluator.model_path}...")
        decode(codec, save_dir, cfg.save_interval)

        # check batches
        dataloader = hydra.utils.instantiate(cfg.data.val)
        data_paths = list(save_dir.glob("batch*.npz"))
        data_paths.sort(key=lambda x: int(x.stem[len("batch") :]))
        for path in data_paths:
            batch, _ = next(dataloader)
            assert (np.load(path)["batch"] == codec.quantize(batch.numpy())).all()

    dist.barrier()
    logger.log("completed")


def encode(
    codec: BitsBackCodec,
    dataloader: Iterator[Tuple[torch.Tensor, dict]],
    num_samples: int,
    save_interval: int,
):
    """
    Encode data using Bits-Back coding.

    Args:
        codec: Codec to use for encoding.
        dataloader: Iterator yielding the data to encode.
        num_samples: Number of samples to encode.
        save_interval: Interval after which to save a checkpoint.
    """
    # raise errors leading to wrong encoding
    np.seterr(all="raise")

    rate_effective: Dict[str, List[np.ndarray]] = {"push": [], "pop": []}
    num_complete = codec.data_shape[0] * codec.data_count * dist.get_world_size()
    while num_complete < num_samples:
        batch, _ = next(dataloader)

        # discretize batch to [0, 2 ** codec.data_prec - 1]
        discr_batch = codec.quantize(batch.numpy())

        # push the discretized batch to the message, and compute rates
        codec.encode(discr_batch)
        stats = codec.statistics()

        if codec.track:
            # check rates (for each diffusion step) and append
            assert stats.rate_effective_pop is not None
            assert stats.rate_effective_push is not None
            assert np.allclose(
                stats.rate_effective_push.sum() - stats.rate_effective_pop.sum(),
                stats.rate_effective,
            )
            rate_effective["push"].append(stats.rate_effective_push)
            rate_effective["pop"].append(stats.rate_effective_pop)

        # log
        stats_dict = stats._asdict()
        stats_dict.pop("rate_effective_push")
        stats_dict.pop("rate_effective_pop")
        logger.logkvs(stats_dict)
        logger.dumpkvs()

        # checkpoint
        num_complete = stats.samples * dist.get_world_size()
        if (
            num_complete >= num_samples
            or stats.samples % save_interval < codec.data_shape[0]
        ):
            ckpt_path = Path(logger.get_dir()) / f"encode-ckpt{stats.samples:06d}.npz"
            np.savez(ckpt_path, **codec.state_dict())

    if codec.track:
        # log rates
        plots = {}
        for name, rate in rate_effective.items():
            out_path = Path(logger.get_dir()) / f"rates-{name}.npz"
            logger.log(f"saving rates-{name} to {out_path}")
            rate_arr = np.stack(rate)
            np.savez(
                out_path,
                mean=rate_arr.mean(axis=0),
                stdd=rate_arr.std(axis=0),
                min=rate_arr.min(axis=0),
                max=rate_arr.max(axis=0),
            )
            plots[name] = plot_evolution(rate_arr, xaxis_title="t", yaxis_title=name)
        wandb.log(plots)


def decode(codec: BitsBackCodec, save_dir: Union[str, Path], save_interval: int):
    """
    Decode data using Bits-Back coding.

    Args:
        codec: Codec to use for decoding.
        save_dir: Directory to save the data.
        save_interval: Interval after which to save a checkpoint.
    """
    save_dir = Path(save_dir)
    while codec.data_count > 0:
        _, data = codec.decode()
        out_path = save_dir / f"batch{codec.data_count:05d}.npz"
        np.savez(out_path, batch=data.astype(np.int32))
        num_remaining = codec.data_shape[0] * codec.data_count
        logger.log(f"{num_remaining} samples remaining...")

        # checkpoint
        if num_remaining % save_interval < codec.data_shape[0]:
            ckpt_path = save_dir / f"decode-ckpt{-num_remaining:06d}.npz"
            np.savez(ckpt_path, **codec.state_dict())


if __name__ == "__main__":
    main()
