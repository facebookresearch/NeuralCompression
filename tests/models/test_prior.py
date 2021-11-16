# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch import Size

from neuralcompression.models import Prior


class TestPrior:
    def test__autoencoder_parameters(self):
        prior = Prior()

        parameters = prior._autoencoder_parameters()

        assert len(parameters) == 14

    def test__bottleneck_optimizer_parameters(self):
        prior = Prior()

        parameters = prior._bottleneck_optimizer_parameters()

        assert len(parameters) == 1

        parameter = parameters[0]

        assert parameter.data.size() == Size([128, 1, 3])

    def test__bottleneck_parameters(self):
        prior = Prior()

        parameters = prior._bottleneck_parameters()

        assert len(parameters) == 1

    def test__intersection_parameters(self):
        prior = Prior()

        parameters = prior._intersection_parameters()

        assert len(parameters) == 0

    def test__optimizer_parameters(self):
        prior = Prior()

        parameters = prior._optimizer_parameters()

        assert len(parameters) == 14

    def test__parameters_dict(self):
        prior = Prior()

        assert len(prior._parameters_dict().keys()) == 15

    def test__union_parameters(self):
        prior = Prior()

        parameters = prior._union_parameters()

        assert len(parameters) == 15

    def test_configure_optimizers(self):
        assert True
