import torch

from neuralcompression.functional import soft_round_conditional_mean


def test_soft_round_conditional_mean():
    for offset in range(-5, 5):
        x = torch.linspace(offset + 0.001, offset + 0.999, 100)

        torch.testing.assert_close(
            torch.round(x),
            soft_round_conditional_mean(x, alpha=5000.0),
            atol=0.001,
            rtol=0.001,
        )
