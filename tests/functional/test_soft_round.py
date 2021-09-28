import torch

from neuralcompression.functional import soft_round


def test_soft_round():
    x = torch.linspace(-2.0, 2.0, 50)

    y = soft_round(x, alpha=1e-13)

    torch.testing.assert_close(x, y)
