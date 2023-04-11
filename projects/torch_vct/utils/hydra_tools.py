# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from omegaconf import OmegaConf

# resolvers: helpers for hydra's configs
OmegaConf.register_new_resolver("eval", lambda x: eval(x))
OmegaConf.register_new_resolver("str", lambda x: str(x))
OmegaConf.register_new_resolver("tuple", lambda x: tuple(x))
OmegaConf.register_new_resolver("prod", lambda x: np.prod(x))
OmegaConf.register_new_resolver("isequal", lambda x, y: x == y)
