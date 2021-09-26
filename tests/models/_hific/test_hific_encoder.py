import torch

from neuralcompression.models._hific._hific_encoder import _HiFiCEncoder


class TestHiFiCEncoder:
    def test_forward(self):
        encoder = _HiFiCEncoder((3, 256, 256))

        x = torch.rand((8, 3, 256, 256))

        assert encoder(x).shape == torch.Size([8, 220, 16, 16])
