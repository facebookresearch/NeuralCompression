# Deep Video Compression

This project trains the 
[Deep Video Compression model (DVC)](https://openaccess.thecvf.com/content_CVPR_2019/html/Lu_DVC_An_End-To-End_Deep_Video_Compression_Framework_CVPR_2019_paper.html) 
on the [Vimeo-90k septuplet](http://toflow.csail.mit.edu/) dataset.

**Note:** A PyTorch implementation linked by the original authors is available
[here](https://github.com/ZhihaoHu/PyTorchVideoCompression).

The project uses 
[PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) 
as a training framework and [Hydra](https://hydra.cc/) for configuration.

## Installation

After installing the `neuralcompression` package following the
[top-level README](https://github.com/facebookresearch/NeuralCompression/README.md)
instructions, install this project's additional dependencies with:

```bash
pip install -r requirements.txt
```

## Training the Model

The config options for this model are documented in `config/base.yaml`. 
The only parameter that must be specified by the user is the path to 
vimeo dataset - this can be done by modifying the config file itself or by 
passing arguments on the command line (see the 
[Hydra documentation](https://hydra.cc/docs/intro#basic-example) for details). 
Training a model locally using default hyperparameters can be run with:

```bash
python train.py data.data_dir=/path/to/vimeo
```

You can modify configuration parameter from the command line - see the
excellent [Hydra documentation](https://hydra.cc/) for details.

### Cluster Training

Using Hydra's 
[submitit plugin](https://hydra.cc/docs/next/plugins/submitit_launcher/), 
you can also launch training jobs on SLURM clusters. 
This can be configured by passing `+mode=submitit_single_node` or 
`+mode=submitit_multi_node` as command line arguments. 
The `--multirun/-m` flag must also be passed to load the plugin. 
For example, to train a model on 2 nodes, each with 3 gpus, run:

```bash
python train.py -m data.data_dir=/path/to/vimeo +mode=submitit_multi_node ngpu=3 trainer.num_nodes=2
```
