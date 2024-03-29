# NeuralCompression

[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/facebookresearch/NeuralCompression/tree/main/LICENSE)
[![Build and Test](https://github.com/facebookresearch/NeuralCompression/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/facebookresearch/NeuralCompression/actions/workflows/build-and-test.yml)

## What's New
- **August 2023 (image compression)** - [Released PyTorch implementation of MS-ILLM](https://github.com/facebookresearch/NeuralCompression/tree/main/projects/illm)
- **April 2023 (video compression)** - [Released PyTorch implementation of VCT](https://github.com/facebookresearch/NeuralCompression/tree/main/projects/torch_vct)
- **November 2022 (image compression)** - [Released Bits-Back coding with diffusion models](https://github.com/facebookresearch/NeuralCompression/tree/main/projects/bits_back_diffusion)!

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
pip install --editable ".[tests]"
```

If you are not interested in matching the test environment, then you can just
apply `pip install -e .`.

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
  - `distributions` - extensions of probability models for compression
  - `functional` - methods for image warping, information cost, flop counting, etc.
  - `layers` - building blocks for compression models
  - `metrics` - `torchmetrics` classes for assessing model performance
  - `models` - complete compression models
  - `optim` - useful optimization utilities

## projects

- `projects` - recipes and code for reproducing papers
  - `bits_back_diffusion` - code for bits-back coding with diffusion models
  - `deep_video_compression` [DVC (Lu et al., 2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Lu_DVC_An_End-To-End_Deep_Video_Compression_Framework_CVPR_2019_paper.html), deprecated
  - `illm` [MS-ILLM (Muckley et al., 2023)](https://proceedings.mlr.press/v202/muckley23a.html)
  - `jax_entropy_coders` - implementations of arithmetic coding and ANS in JAX
  - `torch_vct` [VCT (Mentzer, et al.,)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/54dcf25318f9de5a7a01f0a4125c541e-Abstract-Conference.html)

## Tutorial Notebooks

This repository also features interactive notebooks detailing different 
parts of the package, which can be found in the `tutorials` directory. 
Existing tutorials are:

- Walkthrough of the `neuralcompression` flop counter ([view on Colab](https://colab.research.google.com/github/facebookresearch/NeuralCompression/blob/main/tutorials/Flop_Count_Example.ipynb)).
- Using `neuralcompression.metrics` and `torchmetrics` to calculate rate-distortion curves ([view on Colab](https://colab.research.google.com/github/facebookresearch/NeuralCompression/blob/main/tutorials/Metrics_Example.ipynb)).

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

Model weights released with NeuralCompression are CC-BY-NC 4.0 licensed, as
found in the [WEIGHTS_LICENSE](https://github.com/facebookresearch/NeuralCompression/tree/main/WEIGHTS_LICENSE)
file.

Some of the code may from other repositories and include other licenses.
Please read all code files carefully for details.

## Cite

If you use code for a paper reimplementation. If you would like to also cite
the repository, you can use:

```bibtex
@misc{muckley2021neuralcompression,
    author={Matthew Muckley and Jordan Juravsky and Daniel Severo and Mannat Singh and Quentin Duval and Karen Ullrich},
    title={NeuralCompression},
    howpublished={\url{https://github.com/facebookresearch/NeuralCompression}},
    year={2021}
}
```
