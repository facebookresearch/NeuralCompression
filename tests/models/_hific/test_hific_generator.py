import torch

from neuralcompression.models import HiFiCGenerator


class TestHiFiCGenerator:
    def test_forward(self):
        generator = HiFiCGenerator()

        y = torch.rand((8, 220, 16, 16))

        x_prime = generator(y)

        assert x_prime.shape == torch.Size([8, 3, 256, 256])
