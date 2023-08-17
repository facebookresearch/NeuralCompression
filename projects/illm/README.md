# Official implementation of MS-ILLM

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

## Training

This code makes heavy use of
[hydra](https://github.com/facebookresearch/hydra). Before running the code, it
is recommended to familiarize yourself with `hydra` usage and how to configure
jobs. After this, you will be able to follow the examples below and easily
tailor them to your own uses by modifying the configs.

### Pretraining the HiFiC autoencoder (no GAN)

First, you'll need to pretrain an autoencoder without the GAN.
Populate the configs with the paths to OpenImages V6 and models on your system.
For the initial training stage (without GAN), you need to have the OpenImages
V6 path:

- `conf/data/local_data_openimages.yaml`

Then, launch a 2-GPU job for pretraining the autoencoder at a 0.14 bpp target
bitrate:

```bash
python train.py \
    experiment_name=pretrain0.14bpp \
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

- `conf/pretrained_latent_autoencoder/your_file_name.yaml`

Then, you can run the following:

```bash
python train.py \
    experiment_name=finetune \
    data.batch_size=16 \
    trainer.num_nodes=1 \
    trainer.max_steps=100000 \
    optimizer=model_adamw_disc_const \
    model=hific_autoencoder \
    model.freeze_encoder=true \
    model.freeze_bottleneck=true \
    +pretrained_autoencoder=example \
    distortion_loss=mse_lpips \
    distortion_loss.mse_param=150.0 \
    distortion_loss.lpips_param=1.0 \
    distortion_loss.backbone=alex \
    hydra.launcher.timeout_min=4320 \
    +discriminator=condunet_ch1025_factor8_context220 \
    +lightning_module.generator_weight=0.0005 \
    lightning_module.noisy_context_steps=0 \
    lightning_module.distortion_lam=1.0,0.5,0.25,0.125,0.0625,0.03125,0.015625,0.0078125 \
    +latent_projector=vqvae_xcit_p8_ch64_cb1024_h8 \
    +pretrained_latent_autoencoder=your_file_name
```

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
