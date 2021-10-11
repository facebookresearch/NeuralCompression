import torch
import torch.testing
from torch.distributions import Cauchy, Normal

from neuralcompression.functional import upper_tail


def test_upper_tail():
    torch.testing.assert_close(
        upper_tail(
            Cauchy(torch.tensor([0.0]), torch.tensor([1.0])),
            0.5,
        ),
        torch.tensor([1.0]),
    )

    torch.testing.assert_close(
        upper_tail(
            Normal(torch.tensor([0.0]), torch.tensor([1.0])),
            0.5,
        ),
        torch.tensor([0.6745]),
    )
