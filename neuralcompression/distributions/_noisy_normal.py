from torch import Tensor
from torch.distributions import Normal

from ._uniform_noise import UniformNoise


class NoisyNormal(UniformNoise):
    def __init__(self, **kwargs):
        distribution = Normal(**kwargs)

        super(NoisyNormal, self).__init__(distribution)

    def log_prob(self, value: Tensor) -> Tensor:
        raise NotImplementedError
