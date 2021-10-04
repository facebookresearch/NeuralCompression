import dataclasses

import torch


@dataclasses.dataclass
class _DiscriminatorData:
    authentic_image: torch.Tensor
    synthetic_image: torch.Tensor
    authentic_predictions: torch.Tensor
    synthetic_predictions: torch.Tensor
