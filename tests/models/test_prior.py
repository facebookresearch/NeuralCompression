# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from neuralcompression.models import Prior


class TestPrior:
    def test__autoencoder_parameters(self):
        prior = Prior()

        parameters = prior._autoencoder_parameters()

        assert parameters == set()

    def test__bottleneck_optimizer_parameters(self):
        prior = Prior()

        parameters = prior._bottleneck_optimizer_parameters()

        assert parameters == []

    def test__bottleneck_parameters(self):
        prior = Prior()

        parameters = prior._bottleneck_parameters()

        assert parameters == set()

    def test__intersection_parameters(self):
        prior = Prior()

        parameters = prior._intersection_parameters()

        assert parameters == set()

    def test__optimizer_parameters(self):
        prior = Prior()

        parameters = prior._optimizer_parameters()

        assert parameters == []

    def test__parameters_dict(self):
        prior = Prior()

        parameters = prior._parameters_dict()

        assert parameters == {}

    def test__union_parameters(self):
        prior = Prior()

        parameters = prior._union_parameters()

        assert parameters == set()
