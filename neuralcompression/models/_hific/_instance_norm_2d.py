import torch


def _instance_norm_2d(input_channels, momentum=0.1, affine=True, track_running_stats=False, **kwargs):
    return torch.nn.InstanceNorm2d(
        input_channels,
        momentum=momentum,
        affine=affine,
        track_running_stats=track_running_stats,
    )
