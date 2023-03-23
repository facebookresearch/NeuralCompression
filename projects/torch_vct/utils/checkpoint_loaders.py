# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import sys

import hydra
import torch
import yaml
from model_pipeline import VCTPipeline
from omegaconf import DictConfig


def load_model_checkpoint_from_config(
    path_to_checkpoint: str,
    checkpoint_name: str,
    path_to_config: str,
    device: str = "cpu",
    save_copy_of_ckpt: bool = False,
) -> VCTPipeline:
    """
    Load the model checkpoint specified in 'path_to_checkpoint' using the config
        specified in 'path_to_config'
    """
    try:
        with open(path_to_config, "r") as f:
            model_config = DictConfig(yaml.safe_load(f)["model"]["value"])
    except:
        raise FileNotFoundError(f"File {path_to_config} not found.")

    model = hydra.utils.instantiate(model_config)

    try:
        state_dict = torch.load(
            f"{path_to_checkpoint}/{checkpoint_name}", map_location=torch.device(device)
        )["state_dict"]
    except KeyError:
        state_dict = torch.load(
            f"{path_to_checkpoint}/{checkpoint_name}", map_location=torch.device(device)
        )
    if save_copy_of_ckpt:
        now = datetime.datetime.now()
        path = f"{path_to_checkpoint}/eval_last_{now.date()}_{now.hour}h.pt"
        torch.save(state_dict, path)
    # rm <model.> at the start of each key
    state_dict = {
        k[len("model.") :]: v
        for k, v in state_dict.items()
        if k[: len("model.")] == "model."
    }
    model.load_state_dict(state_dict)
    model.eval()
    if hasattr(model, "bottleneck"):
        model.bottleneck.update()
    elif hasattr(model, "entropy_model"):
        model.entropy_model.update()
    else:
        print("Found no `.update` method", file=sys.stderr)
    return model.to(device)
