"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Approximate the bits/dimension for an image model
by computing the variational lower bound.
Adapted from `improved-diffusion.scripts.image_nll`
(https://github.com/openai/improved-diffusion)
to support hydra and wandb.

Usage (2 GPUs, pretrained model, Cifar10):
1. Local:
    mpiexec -n 2 python compute_vlb.py
2. Slurm:
    python compute_vlb.py -m +mode=submitit hydra.launcher.gpus_per_node=2
"""
from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
import torch
import torch.distributed as dist
import wandb
from bits_back_diffusion.script_util import plot_evolution, setup
from improved_diffusion import dist_util, logger
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="config", config_name="compute_vlb")
def main(cfg: DictConfig):
    """
    Evluate the variational lower bound.

    Args:
        cfg: Configuration for the evaluation.
    """
    model, diffusion, dataloader = setup(cfg)

    logger.log(f"evaluating checkpoint {cfg.evaluator.model_path}")
    num_complete = 0
    bpds = []
    metrics: Dict[str, List[np.ndarray]] = {"vb": [], "mse": [], "xstart_mse": []}

    while num_complete < cfg.evaluator.num_samples:
        batch, model_kwargs = next(dataloader)
        batch = batch.to(dist_util.dev())
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}

        # compute vlb
        batch_metrics = diffusion.calc_bpd_loop(
            model,
            batch,
            clip_denoised=cfg.evaluator.clip_denoised,
            model_kwargs=model_kwargs,
        )

        for key, terms in metrics.items():
            batch_terms = batch_metrics[key]
            if key == "vb":
                batch_terms = torch.cat(
                    [batch_metrics["prior_bpd"].unsqueeze(-1), batch_terms], dim=1
                )
            gathered_terms = [
                torch.zeros_like(batch_terms) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_terms, batch_terms)
            terms.extend([term.cpu().numpy() for term in gathered_terms])

        # the logger does (per default) not average across processes
        total_bpd = batch_metrics["total_bpd"].mean() / dist.get_world_size()
        dist.all_reduce(total_bpd)
        bpds.append(total_bpd.item())
        num_complete += batch.shape[0] * dist.get_world_size()
        logger.logkvs({"total_bpd": np.mean(bpds), "samples": num_complete})
        logger.dumpkvs()

    if dist.get_rank() == 0:
        # log terms
        plots = {}
        for name, terms in metrics.items():
            out_path = Path(logger.get_dir()) / f"{name}-terms.npz"
            logger.log(f"saving {name} terms to {out_path}")
            terms_arr = np.concatenate(terms)[:, ::-1]
            assert terms_arr.shape == (
                num_complete,
                diffusion.num_timesteps + (name == "vb"),
            )
            np.savez(
                out_path,
                mean=terms_arr.mean(axis=0),
                stdd=terms_arr.std(axis=0),
                min=terms_arr.min(axis=0),
                max=terms_arr.max(axis=0),
            )
            plots[name] = plot_evolution(terms_arr, xaxis_title="t", yaxis_title=name)
        wandb.log(plots)

    dist.barrier()
    logger.log("evaluation complete")


if __name__ == "__main__":
    main()
