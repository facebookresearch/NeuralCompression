import dataclasses

import torch


@dataclasses.dataclass
class _IntermediateData:
    authentic_image: torch.Tensor
    synthetic_image: torch.Tensor
    quantized_latent_features: torch.Tensor
    nbpp: float
    qbpp: float
