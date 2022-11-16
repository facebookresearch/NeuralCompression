# Bits-Back coding with diffusion models
> Authored by Julius Berner (https://github.com/juliusberner).

This project provides a lossless encoding scheme for image datasets using
Bits-Back coding and diffusion models.

- Bits-Back coding is based on ['Craystack'](https://github.com/j-towns/craystack), which provides a vectorized version of "Asymmetric Numeral Systems" implemented in NumPy. 
- The diffusion model is based on ['Improved Denoising Diffusion Probabilistic Models'](https://github.com/openai/improved-diffusion), which provides an extension of ['Denoising Diffusion Probabilistic Models'](https://github.com/hojonathanho/diffusion) implemented in PyTorch.
- As exemplary datasets, [Cifar-10](https://www.cs.toronto.edu/~kriz/cifar.html) and [ImageNet](https://image-net.org/)(64x64) is used.

Logging is done using [Weights & Biases](https://wandb.ai) and configuration management is done using [Hydra](https://hydra.cc/).

## Installation

Install this project's dependencies and activate the conda environment with:

```bash
conda env create -f environment.yml
conda activate bits_back_diffusion
```

## Data

### Cifar
Download and prepare the Cifar-10 dataset:
```bash
python datasets/cifar10.py
```

This creates the train and validation datasets at `datasets/cifar10`. This can be changed using `--out_dir=/path/to/data`. However, then you need to add the arguments `data.train.data_dir=/path/to/data/train_32x2` or `data.val.data_dir=/path/to/data/val_32x32` to the commands in section ['Steps'](#steps) below. 

### Imagenet 

Download the downsampled Imagenet(64x64) from Kaggle:
```bash
kaggle datasets download -p datasets/imagenet --unzip ayaroshevskiy/downsampled-imagenet-64x64
```
This requires to sign up for a Kaggle account and to [create an API token](https://github.com/Kaggle/kaggle-api#api-credentials). Further, be aware that unzipping takes some time.

Alternatively, you can also download the dataset from the [Imagenet website](https://image-net.org/download-images.php) and store the images yourself in `datasets/imagenet/train_64x64` and `datasets/imagenet/valid_64x64`. More specifically, you can
* use downsampled images from the pickled batches Train(64x64) part1, Train(64x64) part2, and Val(64x64),
* use the full resolution Imagenet dataset and the automatic downsampling method of the dataloader. 

However, note that due to different downsampling procedures this will yield results different from the ones presented in the paper ['Improved Denoising Diffusion Probabilistic Models'](http://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf). The downsampled Imagenet(64x64) used in the paper seems not to be available anymore on the Imagenet website. If you want to reproduce the results, you should take the version from Kaggle as described above or the version from the [Internet Archive](https://web.archive.org/web/20200410174205/http://www.image-net.org/small/download.php). 


## Pretrained Models

Download pretrained models from the ['Improved Denoising Diffusion Probabilistic Models'](https://github.com/openai/improved-diffusion) repository:

* Cifar-10:
```
curl -o outputs/pretrained/cifar_vlb.pt --create-dirs https://openaipublic.blob.core.windows.net/diffusion/march-2021/cifar10_uncond_vlb_50M_500K.pt
```

* Imagenet(64x64):
```
curl -o outputs/pretrained/imagenet_vlb.pt --create-dirs https://openaipublic.blob.core.windows.net/diffusion/march-2021/imagenet64_uncond_vlb_100M_1500K.pt
```

## Configuration

The config options for this repo are documented in `config`. 
One can modify the config files directly or pass arguments on the command line as is done in the examples below 
(see the [Hydra documentation](https://hydra.cc/docs/intro#basic-example) for details). 

### Weights & Biases

Before starting experiments you need to login to Weights & Biases (wandb):
```bash
wandb login --anonymously
```
You can remove the argument `--anonymously` if you have already created an account on their [website](https://wandb.ai).


### SLURM Cluster 

Using Hydra's 
[submitit plugin](https://hydra.cc/docs/next/plugins/submitit_launcher/), 
you can also launch training/evaluation/compression jobs on SLURM clusters. 
This can be configured by passing `+mode=submitit` and `--multirun/-m` as command line arguments. 

## Steps

In the following `N` denotes the number of GPUs. With 32GiB GPU memory, `N=2` should be a good choice. By default, this uses the Cifar-10 dataset, which can be changed by adding the argument `data=imagenet` to the commands.

### Train (optional)

Train the Denoising Diffusion Probabilistic Model:

* Locally: `mpiexec -n N python train.py`
* Slurm: `python train.py -m +mode=submitit hydra.launcher.gpus_per_node=N`

By default, this will log to the directory `outputs/train/Y-m-d/H-M-S`. You can resume training by appending the argument `hydra.run.dir=outputs/train/Y-m-d/H-M-S`.

### Evaluate (optional)

Evaluate the variational lower bound on the validation dataset:

* Locally: `mpiexec -n N python compute_vlb.py`
* Slurm: `python compute_vlb.py -m +mode=submitit hydra.launcher.gpus_per_node=N`

By defaults, this uses 100 diffusion steps and the pretrained model. This can, e.g., be changed by the arguments `diffusion.timestep_respacing=vlb200` (append number of steps to `vlb`) and `evaluator.model_path=outputs/train/Y-m-d/H-M-S>/.../logging` (directory containing the checkpoints or path to specific checkpoint).

The variational lower bounds should be similar to the results reported in the paper ['Improved Denoising Diffusion Probabilistic Models'](http://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf), see our [results](#results) below.

### Compress

Compress the validation dataset:

* Locally: `python encode.py`
* Slurm: `python encode.py -m +mode=submitit`

As explained above, you can easily change the dataset, the model, and the number of diffusion steps. Also, you can resume encoding by appending the argument `hydra.run.dir=outputs/encode/Y-m-d/H-M-S`.

The effective rate should only be slightly larger than the variational lower bound, see our [results](#results) below. 

## Results

We obtain the following results (in bits-per-dimension) using the pretrained models:

| **Cifar-10** | VLB   | Effective rate |
|--------------|-------|----------------|
| 100 steps    | 3.076 | 3.077          |
| 250 steps    | 3.034 | 3.034          |
| 500 steps    | 3.013 | 3.015          |
| 1000 steps   | 2.990 | 2.992          |
| 2000 steps   | 2.967 | 2.970          |
| 4000 steps   | 2.945 | 2.951          |


| **Imagenet(64x64)**  | VLB   | Effective rate |
|----------------------|-------|----------------|
| 100 steps            | 3.648 | 3.649          |
| 250 steps            | 3.600 | 3.600          |
| 500 steps            | 3.581 | 3.582          |
| 1000 steps           | 3.565 | 3.567          |
| 2000 steps           | 3.553 |                |
| 4000 steps           | 3.544 |                |
