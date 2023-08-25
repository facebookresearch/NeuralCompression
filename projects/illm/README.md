# Official Implementation of MS-ILLM

This is the official code for the following paper:

MJ Muckley, A El-Nouby, K Ullrich, H Jegou, J Verbeek.
[Improving Statistical Fidelity for Neural Image Compression with Implicit Local Likelihood Models](https://proceedings.mlr.press/v202/muckley23a.html).
In *ICML*, 2023.

## Installation

The steps below are recommended with `conda`. If you elect not to use `conda`,
you should be able to create your own install procedure.

Create a new environment with Python 3.10:

```bash
conda create --name illm python=3.10
```

Activate the environment:

```bash
conda activate illm
```

Follow the instructions to install PyTorch (the code is tested with 2.0.0):

https://pytorch.org/get-started/locally/

Install project requirements by running the following in this `illm/`
directory:

```bash
pip install -r requirements.txt
```

## Pretrained Models

**Weights are released under the CC-BY-NC 4.0 license available in the**
**repository root**

This repository is configured to use `torch.hub`. An example for loading the
ILLM model targeting 0.14 bpp is shown below:

```python
import torch

model = torch.hub.load("facebookresearch/NeuralCompression", "msillm_quality_3")
```

To list all available models, you can use:

```python
torch.hub.list("facebookresearch/NeuralCompression")
```

To get information on a model and its target bitrate, you can use:

```python
print(torch.hub.help("facebookresearch/NeuralCompression", "msillm_quality_3"))
```

We release the following models:

| Bitrate   | MS-ILLM            | No-GAN              |
| --------- | ------------------ | ------------------- |
| 0.035 bpp | "msillm_quality_1" | "noganms_quality_1" |
| 0.07 bpp  | "msillm_quality_2" | "noganms_quality_2" |
| 0.14 bpp  | "msillm_quality_3" | "noganms_quality_3" |
| 0.3 bpp   | "msillm_quality_4" | "noganms_quality_4" |
| 0.45 bpp  | "msillm_quality_5" | "noganms_quality_5" |
| 0.9 bpp   | "msillm_quality_6" | "noganms_quality_6" |

The No-GAN models are released for researchers seeking to fine-tune them
with their own methods.

## Reproducing Results

The `eval_folder_example.py` script provides an example for reproducing the
numbers from the paper. Simply run

```bash
python eval_folder_example.py $PATH_TO_CLIC2020
```

And it should run on the folder you provide and print the metrics for the
0.14 bpp target model.

## Training

This code makes heavy use of
[hydra](https://github.com/facebookresearch/hydra). Before running the code, it
is recommended to familiarize yourself with `hydra` usage and how to configure
jobs. After this, you will be able to follow the examples below and easily
tailor them to your own uses by modifying the configs.

### Pretraining the HiFiC autoencoder (no GAN)

First, you'll need to pretrain an autoencoder without the GAN.
Launch a 2-GPU job for pretraining the autoencoder at a 0.14 bpp target
bitrate:

```bash
python train.py \
    experiment_name=pretrain0.14bpp \
    data.open_images_root=$PATH_TO_OPENIMAGES \
    data.batch_size=8 \
    distortion_loss=mse_lpips \
    distortion_loss.mse_param=150.0 \
    distortion_loss.lpips_param=1.0 \
    distortion_loss.backbone=alex \
    optimizer.model_opt.lr=0.0003 \
    rate_target=target-0.14 \
    model=hific_autoencoder
```

**Note:** If you're in a SLURM cluster environment and want to train all rates,
you can use the multirun feature of `hydra`. An example SLURM config is in
`conf/mode/submitit_multi_node.yaml`.

## Fine-tuning the HiFiC autoencoder (with GAN)

For the second training stage (with GAN), you'll need to update the path to
your pretrained HiFiC autoencoder by swapping out the `path` field in this
file:

- `conf/pretrained_autoencoder/example.yaml`

And you'll need to put your pretrained VQ-VAE here:

- `conf/pretrained_latent_autoencoder/example.yaml`

Alternatively, you can use the pretrained autoencoder and VQ-VAE from'
`torch.hub` with the `20221108_vqvae_xcit_p8_ch64_cb1024_h8.yaml` and 
`nogan_target0.14bpp` configs.:

```bash
python train.py \
    experiment_name=finetune \
    data.open_images_root=/datasets01/open_images/030822/v6 \
    data.batch_size=16 \
    trainer.max_steps=100000 \
    optimizer=model_adamw_disc_const \
    model=hific_autoencoder \
    model.freeze_encoder=true \
    model.freeze_bottleneck=true \
    distortion_loss=mse_lpips \
    distortion_loss.mse_param=150.0 \
    distortion_loss.lpips_param=1.0 \
    distortion_loss.backbone=alex \
    +discriminator=condunet_ch1025_factor8_context220 \
    +lightning_module.generator_weight=0.008 \
    +latent_projector=vqvae_xcit_p8_ch64_cb1024_h8 \
    +pretrained_autoencoder=nogan_target0.14bpp \
    +pretrained_latent_autoencoder=20221108_vqvae_xcit_p8_ch64_cb1024_h8
```

This should give a model with about a 2.3 validation FID on OpenImages V6.

## Other notes

### Logging

By default the project uses [Weights & Biases](https://wandb.ai/site) for
logging. If you don't have `wandb` installed, the trainer should try to
fall-back to training without it (note: images will not be logged.), but there
may be some issues you'll have to debug. Fortunately, all of the logging code
is confined to `train.py`. You shouldn't have to modify anything outside of
that file.

If you're using `wandb`, you'll need to ensure that you're logged into your
account or you've set up the anonymous logging mode.

### Compressing autoencoder

The paper was written with the HiFiC architecture, and this is used in all
experiments. The nice thing about the HiFiC architecture is that it's very
stable for discriminator fine-tuning. For newer architectures, such as ELIC,
discriminator fine-tuning can be more difficult. It is possible to get good
results, but it requires more hyperparameter tuning, and you may need to use
different weights or learning rates for different parts of the R-D curve.

If you find recipes that work particularly well, feel free to post them as a
Discussion (or go ahead and publish and toss a cite :) ).

## BibTeX
```bibtex
@inproceedings{muckley2023improving,
  author = {Muckley, Matthew J. and El-Nouby, Alaaeldin and Ullrich, Karen and Jégou, Hervé and Verbeek, Jakob},
  title = {Improving Statistical Fidelity for Neural Image Compression with Implicit Local Likelihood Models},
  booktitle = {International Conference on Machine Learning},
  year = {2023},
}
```
