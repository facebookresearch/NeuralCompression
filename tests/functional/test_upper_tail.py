from torch import tensor
from torch.distributions import Cauchy, Normal
from torch.testing import assert_close

from neuralcompression.functional import upper_tail


def test_upper_tail():
    assert_close(
        upper_tail(
            Cauchy(tensor([0.0]), tensor([1.0])),
            0.5,
        ),
        tensor([1.0]),
    )

    assert_close(
        upper_tail(
            Normal(tensor([0.0]), tensor([1.0])),
            0.5,
        ),
        tensor([0.6745]),
    )
