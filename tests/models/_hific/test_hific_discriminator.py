import torch

from neuralcompression.models._hific._hific_discriminator import _HiFiCDiscriminator


class TestHiFiCDiscriminator:
    def test_forward(self):
        discriminator = _HiFiCDiscriminator()

        x = torch.rand((8, 3, 256, 256))
        y = torch.rand((8, 220, 16, 16))

        a, b = discriminator(x, y)

        assert a.shape == torch.Size([2048, 1])
        assert b.shape == torch.Size([2048, 1])
