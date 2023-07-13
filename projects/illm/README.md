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

**NOTE:** This code uses `wandb` for metrics logging. Most of the code is
logger agnostic, but the image callback in `train.py` does include some
`wandb`-specific code. If you would like to use a logger other than `wandb`,
you can modify or remove the image callback and still have all metrics
(just no images).

If you're using `wandb`, you'll need to ensure that you're logged into your
account or you've set up the anonymous logging mode.

## Training

Populate the configs with the paths to OpenImages V6 and models on your system.
For the initial training stage (without GAN), you need to have the OpenImages
V6 path:

- `conf/data/local_data_openimages.yaml`

For the second training stage (with GAN), you'll need to update the path to
your pretrained HiFiC autoencoder here:

- `conf/pretrained_autoencoder/example.yaml`

And you'll need to put your pretrained VQ-VAE here:

- `conf/pretrained_latent_autoencoder/your_file_name.yaml`

