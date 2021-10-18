import torch
import torch.testing

from neuralcompression.layers import NonNegativeParameter


class TestNonNegativeParameter:
    def test___init__(self):
        x = torch.rand((4,))

        non_negative_parameter = NonNegativeParameter(x)

        assert non_negative_parameter.initial_value.shape == x.shape

        torch.testing.assert_close(
            non_negative_parameter.initial_value,
            torch.sqrt(torch.max(x, x - x)),
        )

    def test_parameterize(self):
        x = torch.rand((4,))

        non_negative_parameter = NonNegativeParameter(
            shape=x.shape,
            dtype=torch.float,
        )

        parameterized = non_negative_parameter.parameterize(x)

        assert parameterized.shape == x.shape

        torch.testing.assert_equal(
            parameterized,
            torch.sqrt(torch.max(x, x - x)),
        )

    def test_forward(self):
        x = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])

        minimum = 0.2

        non_negative_parameter = NonNegativeParameter(
            minimum=minimum,
            shape=x.shape,
            dtype=torch.float,
        )

        reparameterized = non_negative_parameter(x)

        assert reparameterized.shape == x.shape

        torch.testing.assert_close(
            float(reparameterized.min()),
            0.01,
        )
