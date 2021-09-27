import torch

from neuralcompression.models._hific._hific_generator import _HiFiCGenerator


class TestHiFiCGenerator:
    def test_forward(self):
        generator = _HiFiCGenerator((16, 16, 16), 8)

        y = torch.rand((8, 16, 16, 16))

        x_prime = generator(y)

        assert x_prime.shape == torch.Size([8, 3, 256, 256])
