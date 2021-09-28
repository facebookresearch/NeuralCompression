import torch

from neuralcompression.functional import soft_round_inverse


def test_soft_round_inverse():
    x = torch.linspace(-2.0, 2.0, 50)

    y = soft_round_inverse(x, alpha=1e-13)

    torch.testing.assert_close(x, y)
