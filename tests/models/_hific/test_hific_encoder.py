import torch

from neuralcompression.models._hific import HiFiCEncoder


class TestHiFiCEncoder:
    def test_forward(self):
        encoder = HiFiCEncoder((3, 256, 256))

        x = torch.rand((8, 3, 256, 256))

        assert encoder(x).shape == torch.Size([8, 220, 16, 16])
