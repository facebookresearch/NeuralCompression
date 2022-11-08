"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Parts of the code are based on the repository
https://github.com/openai/improved-diffusion.
"""
import copy
import hashlib
from pathlib import Path
from typing import List, Set, Union

import numpy as np
import plotly.graph_objs as go
import torch
import torch.distributed as dist
import wandb
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
from improved_diffusion import dist_util
from improved_diffusion import gaussian_diffusion as gd
from improved_diffusion import logger, respace, train_util
from omegaconf import DictConfig, OmegaConf


def divide_by_n_proc(x: int):
    """
    Divide the input by the world size.

    Args:
        x: Input to divide.
    Returns:
        Quotient of input and world size.
    """
    quotient, remainder = divmod(x, dist.get_world_size())
    if remainder:
        raise ValueError(
            f"{x} is not divisible by the number of processes {dist.get_world_size()}."
        )
    return quotient


OmegaConf.register_new_resolver("divide_by_n_proc", divide_by_n_proc)


class WandbOutputFormat(logger.KVWriter):
    """
    Weights & Biases logger.

    Args:
        dir: Directory for saving the logs.
    """

    def __init__(self, dir: str):
        save_dir = Path(dir).absolute()
        save_dir.mkdir(parents=True, exist_ok=True)
        self.dir = str(save_dir)

        # construct SHA256 hash of the log dir for the wandb id
        sha = hashlib.sha256()
        sha.update(self.dir.encode())

        run_dir = Path(HydraConfig.get().run.dir)
        wandb.require(experiment="service")
        self.run = wandb.init(
            dir=self.dir,
            name="/".join(
                [
                    run_dir.parent.name,
                    run_dir.name,
                ]
            ),
            project="bits-back-diffusion",
            group=HydraConfig.get().job.name,
            tags=[t[-64:] for t in HydraConfig.get().overrides.task],
            id=sha.hexdigest(),
            resume="allow",
        )

    def writekvs(self, kvs: dict):
        """
        Log key/value pairs to Weights & Biases.

        Args:
            kvs: Dictionary of key/value pairs.
        """
        kvs = {k.replace("_", "/"): v for k, v in kvs.items()}
        if self.run:
            self.run.log(kvs)
        else:
            raise RuntimeError(" Weights&Biases run not available.")

    def close(self):
        """
        Close the logger.
        """
        if self.run:
            self.run.finish()
            self.run = None


def parse_resume_step_from_filename(filename: Union[str, Path]) -> int:
    """
    Parse filenames of the form path/to/modelNNNNNN.pt or
    path/to/ema_0.xxxx_NNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.

    Adapted from `improved-diffusion.train_util.parse_resume_step_from_filename`
    (https://github.com/openai/improved-diffusion)
    to parse checkpoints of exponential moving averages and codecs.

    Args:
        filename: Path of the model/ema checkpoint.
    Returns:
        The resume step of the checkpoint.
    """
    path = Path(filename)
    assert path.suffix in [".pt", ".npz"]
    assert any(s in path.name for s in ["ema", "model", "encode", "decode"])
    return int(path.stem[-6:])


def update_checkpoint(
    cfg: DictConfig,
    key: str,
    prefix: str = "model",
    suffix: str = "pt",
):
    """
    Udpdate the checkpoint in a configuration.

    Args:
        cfg: The configuration to update.
        key: The dot-notation key pointing to the checkpoint
            in the configuration.
        prefix: File extension of the checkpoint file.
        suffix: Suffix of the checkpoint file.
    """
    ckpt_path = OmegaConf.select(cfg, key)

    if ckpt_path:
        ckpt_path = Path(to_absolute_path(ckpt_path))
    else:
        assert logger.get_dir() is not None
        ckpt_path = Path.cwd() / logger.get_dir()

    if ckpt_path.is_dir():
        # search for the last checkpoint
        ckpt_files = list(ckpt_path.glob(f"{prefix}*.{suffix}"))
        if ckpt_files:
            ckpt_path = max(ckpt_files, key=parse_resume_step_from_filename)

    if ckpt_path.is_file():
        OmegaConf.update(cfg, key, str(ckpt_path))


def setup(
    cfg: DictConfig,
    train: bool = False,
    log_dir: str = "logging",
    log_ranks: bool = False,
):
    """
    Set up an experiment.

    Args:
        cfg: Configuration to use for setting up the experiment.
        train: Whether to use training or evaluation configurations.
        log_dir: Directory for logging.
            If `log_ranks` is True, the rank will be added as suffix.
        log_ranks: Whether to log each process separately.

    Returns:
        The model, the diffusion, and an iterator yielding the data.
    """
    # distributed and logging setup
    dist_util.setup_dist()
    if log_ranks:
        log_dir += f"_rank{dist.get_rank():03d}"
    logger.configure(dir=log_dir, format_strs=["stdout", "log", "csv"])

    # update configs
    data = cfg.data.train if train else cfg.data.val
    data.data_dir = to_absolute_path(data.data_dir)
    if train:
        update_checkpoint(cfg, "trainer.resume_checkpoint")
    else:
        update_checkpoint(cfg, "evaluator.model_path", prefix="ema")

    # wandb setup
    if log_ranks or (dist.get_rank() == 0):
        current_logger = logger.get_current()
        if current_logger:
            wandb_output_format = WandbOutputFormat(current_logger.dir)
            current_logger.output_formats.append(wandb_output_format)
            wandb.config.update(
                OmegaConf.to_container(cfg, resolve=True), allow_val_change=True
            )

    logger.log("creating model and diffusion...")
    model = instantiate(cfg.model)
    if not train:
        model.load_state_dict(
            dist_util.load_state_dict(cfg.evaluator.model_path, map_location="cpu")
        )
        eff_batch_size = data.batch_size * dist.get_world_size()
        if cfg.evaluator.num_samples % eff_batch_size:
            logger.log(
                f"dropping samples as number of samples {cfg.evaluator.num_samples}"
                f" is not divisible by effective batchsize {eff_batch_size}."
            )

    model.to(dist_util.dev())
    model.train(train)
    diffusion = instantiate(cfg.diffusion)

    logger.log(f"creating data loader from {data.data_dir}...")
    dataloader = instantiate(data)

    return model, diffusion, dataloader


class TrainLoop(train_util.TrainLoop):
    """
    Adapted from `improved_diffusion.train_util.TrainLoop`
    (https://github.com/openai/improved-diffusion)
    to remove MPI broadcasts for parameter loading.
    """

    def _load_and_sync_parameters(self):
        if self.resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(self.resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(
                    f"loading model from checkpoint: {self.resume_checkpoint}..."
                )
                self.model.load_state_dict(
                    torch.load(self.resume_checkpoint, map_location=dist_util.dev())
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate: float) -> list:
        ema_params = copy.deepcopy(self.master_params)

        ema_checkpoint = train_util.find_ema_checkpoint(
            self.resume_checkpoint, self.resume_step, rate
        )
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = torch.load(ema_checkpoint, map_location=dist_util.dev())
                ema_params = self._state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params


def space_timesteps(
    num_timesteps: int, section_counts: Union[List[int], str]
) -> Set[int]:
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    Adapted from `improved_diffusion.respace.space_timesteps`
    (https://github.com/openai/improved-diffusion) to add "vlbN" method.

    Args:
        num_timesteps: The number of diffusion steps in the original
                          process to divide up.
        section_counts: Either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special cases, use "vlbN" or
                           "ddimN" where N is a number of steps to use a dense
                           inital striding or the striding from the DDIM paper.
    Returns:
        The subset of diffusion steps.
    """
    all_steps: List[int] = []

    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    logger.log(f"respaced to {desired_count} steps...")
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        if section_counts.startswith("vlb"):
            desired_count = int(section_counts[len("vlb") :])
            if (num_timesteps // desired_count) < desired_count:
                all_steps += list(range(num_timesteps // desired_count))
                section_counts = [desired_count - num_timesteps // desired_count + 1]
            else:
                raise ValueError("please select more steps.")
        else:
            section_counts = [int(x) for x in section_counts.split(",")]

    start_idx = 0
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1.0
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    steps = set(all_steps)
    logger.log(f"respaced to {len(steps)} steps...")
    return steps


def create_gaussian_diffusion(
    steps: int = 1000,
    learn_sigma: bool = False,
    sigma_small: bool = False,
    noise_schedule: str = "linear",
    use_kl: bool = False,
    predict_xstart: bool = False,
    rescale_timesteps: bool = False,
    rescale_learned_sigmas: bool = False,
    timestep_respacing: Union[List[int], str] = "",
) -> respace.SpacedDiffusion:
    """
    Adapted from `improved_diffusion.train_util.create_gaussian_diffusion`
    (https://github.com/openai/improved-diffusion)
    to replace the `space_timesteps` function.
    """
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return respace.SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def plot_evolution(data: np.ndarray, **kwargs) -> go.Figure:
    """
    Plot data over steps.

    Args:
        data: Data of shape [steps, iterations] or [steps] to plot.
            In the first case mean and stdd over steps are plotted.
        kwargs: Keyword arguments to update the figure layout.
    Returns:
        The plotly figure of the plot.
    """

    mean = data
    if data.ndim > 1:
        mean = data.mean(axis=0)

    x = list(range(len(mean)))
    plot_data = [
        go.Scatter(
            name="mean",
            x=x,
            y=mean,
            mode="lines",
        )
    ]

    if data.ndim > 1:
        stdd = data.std(axis=0)
        x_round = np.append(x, x[::-1])
        plot_data += [
            go.Scatter(
                x=x_round,
                y=np.append(mean + stdd, (mean - stdd)[::-1]),
                fill="toself",
                fillcolor="rgba(0,100,80,0.4)",
                line=dict(color="rgba(255,255,255,0)"),
                name="stdd",
            ),
            go.Scatter(
                x=x_round,
                y=np.append(data.max(axis=0), data.min(axis=0)[::-1]),
                fill="toself",
                fillcolor="rgba(68, 68, 68, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                name="range",
            ),
        ]

    fig = go.Figure(plot_data)
    fig.update_layout(**kwargs, hovermode="x")
    return fig
