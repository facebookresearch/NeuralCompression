import numpy
import torch
import torch.testing

from neuralcompression.functional import lower_bound


def test_lower_bound():
    rng = numpy.random.default_rng(0xFEEEFEEE)

    x = torch.tensor(rng.random((4,)), dtype=torch.float, requires_grad=True)

    (bound,) = rng.random(1)

    y = lower_bound(x, bound, gradient="disconnected")

    torch.testing.assert_equal(y, torch.clamp_max(x, bound))

    y.backward(x)

    assert x.grad is not None

    torch.testing.assert_equal(x.grad, (x >= bound) * x)

    x = torch.tensor(rng.random((4,)), dtype=torch.float, requires_grad=True)

    (bound,) = rng.random(1)

    y = lower_bound(x, bound, gradient="identity")

    torch.testing.assert_equal(y, torch.clamp_max(x, bound))

    y.backward(x)

    assert x.grad is not None

    x = torch.tensor(rng.random((4,)), dtype=torch.float, requires_grad=True)

    (bound,) = rng.random(1)

    y = lower_bound(x, bound, gradient="identity_if_towards")

    torch.testing.assert_equal(y, torch.clamp_max(x, bound))

    y.backward(x)

    assert x.grad is not None

    torch.testing.assert_equal(x.grad, (x >= bound) * x)
