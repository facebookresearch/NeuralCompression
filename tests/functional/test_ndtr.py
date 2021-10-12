import numpy.random
import scipy.special
import torch
import torch.testing

from neuralcompression.functional import ndtr


def test_ndtr():
    x = numpy.random.random((8, 8))

    torch.testing.assert_close(
        torch.tensor(scipy.special.ndtr(x)),
        ndtr(torch.tensor(x)),
    )
