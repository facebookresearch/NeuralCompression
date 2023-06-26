# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn


class Discriminator(nn.Module):
    @property
    def is_conditional(self) -> bool:
        raise NotImplementedError
