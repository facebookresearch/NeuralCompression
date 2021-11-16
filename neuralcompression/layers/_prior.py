# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, OrderedDict

import torch
import torch.nn.init
from compressai.entropy_models import EntropyBottleneck
from torch import Tensor
from torch.nn import (
    Conv2d,
    ConvTranspose2d,
    Module,
)


class Prior(Module):
    def __init__(
        self,
        n: int,
        m: int,
    ):
        super(Prior, self).__init__()

        self._n = n
        self._m = m

        self.entropy_coding = EntropyBottleneck(self._m)

        for module in self.modules():
            if isinstance(module, (Conv2d, ConvTranspose2d)):
                torch.nn.init.kaiming_normal_(module.weight)

                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    @property
    def _bottleneck_loss(self) -> float:
        losses = []

        for module in self.modules():
            if isinstance(module, EntropyBottleneck):
                losses += [module.loss()]

        return sum(losses)

    def update(self, force: bool = False) -> bool:
        updated = False

        for module in self.children():
            if not isinstance(module, EntropyBottleneck):
                continue

            updated |= module.update(force=force)

        return updated

    def load_state_dict(
        self,
        state_dict: OrderedDict[str, Tensor],
        strict: bool = True,
    ):
        self._update_registered_buffers(
            self.entropy_coding,
            "entropy_coding",
            [
                "_quantized_cdf",
                "_offset",
                "_cdf_length",
            ],
            state_dict,
        )

        super(Prior, self).load_state_dict(state_dict, strict)

    @staticmethod
    def _update_registered_buffers(
        module: Module,
        module_name: str,
        buffer_names: List[str],
        state_dict: OrderedDict,
        policy: str = "resize_if_empty",
        dtype: torch.dtype = torch.int32,
    ):
        valid_buffer_names = []

        for name, _ in module.named_buffers():
            valid_buffer_names += [name]

        for buffer_name in buffer_names:
            size = state_dict[f"{module_name}.{buffer_name}"].size()

            registered_buffers = []

            for name, buffer in module.named_buffers():
                if name == buffer_name:
                    registered_buffers += [buffer]

            registered_buffer: Optional[Tensor]

            if registered_buffers:
                registered_buffer = registered_buffers[0]
            else:
                registered_buffer = None

            if policy in ("resize_if_empty", "resize"):
                if policy == "resize" or registered_buffer.numel() == 0:
                    registered_buffer.resize_(size)
            elif policy == "register":
                module.register_buffer(buffer_name, torch.zeros(size, dtype=dtype))
