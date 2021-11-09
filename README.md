# NeuralCompression

[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/facebookresearch/NeuralCompression/tree/main/LICENSE)
[![Build and Test](https://github.com/facebookresearch/NeuralCompression/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/facebookresearch/NeuralCompression/actions/workflows/build-and-test.yml) [![Documentation Status](https://readthedocs.org/projects/neuralcompression/badge/?version=latest)](https://neuralcompression.readthedocs.io/en/latest/?badge=latest)

## What's New

- **July 2021 (image compression)** - [Released implemenation of Scale Hyperprior](https://github.com/facebookresearch/NeuralCompression/tree/main/projects/scale_hyperprior_lightning)
- **July 2021 (video compression)** - [Released implementation of DVC](https://github.com/facebookresearch/NeuralCompression/tree/main/projects/deep_video_compression)

## About

NeuralCompression is a Python repository dedicated to research of neural
networks that compress data. The repository includes tools such as JAX-based
entropy coders, image compression models, video compression models, and metrics
for image and video evaluation.

NeuralCompression is alpha software. The project is under active development.
The API will change as we make releases, potentially breaking backwards
compatibility.

## Installation

NeuralCompression is a project currently under development. You can install the
repository in development mode.

### PyPI Installation

First, install PyTorch according to the directions from the
[PyTorch website](https://pytorch.org/). Then, you should be able to run

```bash
pip install neuralcompression
```

to get the latest version from PyPI.

### Development Installation

First, clone the repository and navigate to the NeuralCompression root
directory and install the package in development mode by running:

```bash
pip install --editable ".[dev, docs]"
```

If you are not interested in matching the test environment, then you only need
to apply the second step to install.

## Repository Structure

We use a 2-tier repository structure. The `neuralcompression` package contains
a core set of tools for doing neural compression research. Code committed to
the core package requires stricter linting, high code quality, and rigorous
review. The `projects` folder contains code for reproducing papers and training
baselines. Code in this folder is not linted aggressively, we don't enforce
type annotations, and it's okay to omit unit tests.

The 2-tier structure enables rapid iteration and reproduction via code in
`projects` that is built on a backbone of high-quality code in
`neuralcompression`.

## neuralcompression

- `neuralcompression` - base package
  - `data` - PyTorch data loaders for various data sets
  - `entropy_coders` - lossless compression algorithms in JAX
    - `craystack` - an implementation of the rANS algorithm with the
    [craystack](https://github.com/j-towns/craystack) API
  - `functional` - methods for image warping, information cost, flop counting, etc.
  - `layers` - building blocks for compression models
  - `metrics` - `torchmetrics` classes for assessing model performance
  - `models` - complete compression models

## projects

- `projects` - recipes and code for reproducing papers
  - `scale_hyperprior_lightning` [Scale Hyperprior (Balle et al., 2018)](https://arxiv.org/abs/1802.01436)
  - `deep_video_compression` [DVC (Lu et al., 2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Lu_DVC_An_End-To-End_Deep_Video_Compression_Framework_CVPR_2019_paper.html)

## Tutorial Notebooks

This repository also features interactive notebooks detailing different 
parts of the package, which can be found in the `tutorials` directory. 
Existing tutorials are:

- Walkthrough of the `neuralcompression` flop counter ([view on Colab](https://colab.research.google.com/github/facebookresearch/NeuralCompression/blob/main/tutorials/Flop_Count_Example.ipynb)).
- Using `neuralcompression.metrics` and `torchmetrics` to calculate rate-distortion curves ([view on Colab](https://colab.research.google.com/github/facebookresearch/NeuralCompression/blob/main/tutorials/Metrics_Example.ipynb)).

## Getting Started

For an example of package usage, see the
[Scale Hyperprior](https://github.com/facebookresearch/NeuralCompression/tree/main/projects/scale_hyperprior_lightning) for an example of how
to train an image compression model in PyTorch Lightning. See
[DVC](https://github.com/facebookresearch/NeuralCompression/tree/main/projects/deep_video_compression) for a video compression example.

## Contributions

Please read our [CONTRIBUTING](https://github.com/facebookresearch/NeuralCompression/tree/main/.github/CONTRIBUTING.md) guide and our
[CODE_OF_CONDUCT](https://github.com/facebookresearch/NeuralCompression/tree/main/.github/CODE_OF_CONDUCT.md) prior to submitting a pull
request.

We test all pull requests. We rely on this for reviews, so please make sure any
new code is tested. Tests for `neuralcompression` go in the `tests` folder in
the root of the repository. Tests for individual projects go in those projects'
own `tests` folder.

We use `black` for formatting, `isort` for import sorting, `flake8` for
linting, and `mypy` for type checking.

## License

NeuralCompression is MIT licensed, as found in the [LICENSE](https://github.com/facebookresearch/NeuralCompression/tree/main/LICENSE) file.

## Cite

If you find NeuralCompression useful in your work, feel free to cite

```bibtex
@misc{muckley2021neuralcompression,
    author={Matthew Muckley and Jordan Juravsky and Daniel Severo and Mannat Singh and Quentin Duval and Karen Ullrich},
    title={NeuralCompression},
    howpublished={\url{https://github.com/facebookresearch/NeuralCompression}},
    year={2021}
}
```
