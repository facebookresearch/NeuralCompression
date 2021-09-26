import torch

from neuralcompression.models._hific._hific_generator import _HiFiCGenerator


class TestHiFiCGenerator:
    def test_forward(self):
        generator = _HiFiCGenerator((16, 16, 16), 8)

        x = torch.rand((8, 16, 16, 16))

        assert generator(x).shape == torch.Size([8, 3, 256, 256])
