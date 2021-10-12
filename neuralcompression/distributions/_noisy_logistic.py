from torch import Tensor
from torch.distributions import (
    AffineTransform,
    SigmoidTransform,
    TransformedDistribution,
    Uniform,
)

from ._uniform_noise import UniformNoise


class NoisyLogistic(UniformNoise):
    def __init__(self, **kwargs):
        distribution = TransformedDistribution(
            Uniform(0, 1),
            [SigmoidTransform().inv, AffineTransform(**kwargs)],
        )

        super(NoisyLogistic, self).__init__(distribution)

    def log_prob(self, value: Tensor) -> Tensor:
        raise NotImplementedError