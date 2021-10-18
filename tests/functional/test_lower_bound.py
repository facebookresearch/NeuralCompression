import numpy
import torch
import torch.testing

from neuralcompression.functional import lower_bound


def test_lower_bound():
    rng = numpy.random.default_rng(0xFEEEFEEE)

    x = torch.tensor(rng.random((4,)), dtype=torch.float, requires_grad=True)

    (bound,) = rng.random(1)

    y = lower_bound(x, bound)

    torch.testing.assert_equal(y, torch.clamp_max(x, bound))

    y.backward(x)

    assert x.grad is not None

    torch.testing.assert_equal(x.grad, (x >= bound) * x)
