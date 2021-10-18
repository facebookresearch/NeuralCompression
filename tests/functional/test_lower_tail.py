import torch
import torch.testing
from torch.distributions import Normal, Uniform

from neuralcompression.functional import lower_tail


def test_lower_tail():
    torch.testing.assert_close(
        lower_tail(
            Normal(0.0, 1.0),
            0.5,
        ),
        torch.tensor([-0.6745]),
    )

    torch.testing.assert_close(
        lower_tail(
            Uniform(0.0, 1.0),
            0.5,
        ),
        torch.tensor([0.25]),
    )
