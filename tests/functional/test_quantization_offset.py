from torch.distributions import Gamma, Laplace, Normal

from neuralcompression.functional import quantization_offset


def test_quantization_offset():
    assert quantization_offset(Gamma(5.0, 1.0)) == 0.0

    assert quantization_offset(Laplace(-2.0, 5.0)) == 0.0

    assert quantization_offset(Normal(3.0, 5.0)) == 0.0
