# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import sys
import os
from pytorch_lightning import Trainer
from omegaconf import DictConfig

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from datamodules.uvg import UVGDataModule
from model_lightning import VCTModule

from utils.datavis import show_video
from utils.checkpoint_loaders import load_model_checkpoint_from_config
import matplotlib.pyplot as plt


# %%
DEBUG = False
DEVICE = "cuda:0"  # "cpu"

# %%
num_frames = 60  # number in frames per batch
save_copy_of_ckpt = False

if DEBUG:
    print("Debug Mode!")
    num_frames = 10
    save_copy_of_ckpt = False

uvg = UVGDataModule(
    data_dir="/checkpoint/desi/datasets/uvg", num_frames=num_frames, num_workers=10
)

# %%
# test = uvg.test_dataloader()
# sample = next(iter(test))

# B, T, _, H, W = sample.shape
# num_pixels = B * T * H * W

# show_video(sample.video_tensor[0][:10, ...].permute(0, 2, 3, 1))
# plt.show()


# %%
run_name = "20230309_lion_VCT_official_0.01"  # "20230306_VCT_official"
usr = "karenu"  # "desi"  #
run_ids = [13]  # , 0, 1]
print(run_ids)
ckpt = "last.ckpt"  # "epoch=0-step=218000.ckpt"  #
for i in run_ids:
    print(f"RUN ID: {i}, num_frames: {num_frames}, store_ckp: {save_copy_of_ckpt}")
    model = load_model_checkpoint_from_config(
        path_to_checkpoint=f"/checkpoint/{usr}/torch_vct/experiments/{run_name}/{i}/checkpoints",
        checkpoint_name=f"{ckpt}",
        path_to_config=f"/checkpoint/{usr}/torch_vct/experiments/{run_name}/{i}/wandb_logs/wandb/latest-run/files/config.yaml",
        device=DEVICE,
        save_copy_of_ckpt=save_copy_of_ckpt,
    )
    modelmodule = VCTModule(model, DictConfig({}))
    if hasattr(modelmodule.model.analysis_transform, "_max_frames"):
        modelmodule.model.analysis_transform._max_frames = 5  # type: ignore
        modelmodule.model.synthesis_transform._max_frames = 5  # type: ignore
    trainer = Trainer(accelerator="gpu", devices=1, strategy="ddp")
    print("STARTING EVAL")
    res = trainer.test(modelmodule, datamodule=uvg)
    print(res)
