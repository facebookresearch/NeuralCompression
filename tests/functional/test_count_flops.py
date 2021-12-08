# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn

from neuralcompression.functional import count_flops
from neuralcompression.models import ScaleHyperprior


@pytest.mark.parametrize(
    "batch_size, dim_in, dim_out",
    [(1, 1, 1), (2, 1, 10), (3, 10, 10), (4, 10, 1), (5, 6, 7)],
)
def test_flop_count_ffn(batch_size, dim_in, dim_out):
    # Testing the flop counter counts simple modules as expected.

    ffn_model = torch.nn.Sequential(
        torch.nn.Linear(dim_in, dim_out, bias=False),
        torch.nn.Linear(dim_out, dim_out, bias=False),
    )
    ffn_inputs = (torch.randn(batch_size, dim_in),)

    total_flops, _, unsupported_ops = count_flops(ffn_model, ffn_inputs)
    assert total_flops == batch_size * dim_in * dim_out + batch_size * dim_out * dim_out
    assert len(unsupported_ops) == 0


@pytest.mark.parametrize(
    "input_shape, layer_channels, layer_strides, layer_kernels, layer_padding",
    (
        ((1, 3, 64, 64), (10, 6, 11), (1, 2, 1), (1, 2, 3), (0, 1, 5)),
        ((3, 10, 11, 127), (2, 4, 8, 1), (2, 2, 2, 2), (4, 7, 1), (0, 2, 3, 0)),
    ),
)
def test_flop_count_conv(
    input_shape, layer_channels, layer_strides, layer_kernels, layer_padding
):
    # Testing that the flop counter counts a convolutional model as expected
    # (i.e. when compared to a reference value).

    class ConvModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList()
            cin = input_shape[1]

            for cout, stride, kernel, padding in zip(
                layer_channels, layer_strides, layer_kernels, layer_padding
            ):
                self.layers.append(
                    torch.nn.Conv2d(
                        cin, cout, kernel_size=kernel, stride=stride, padding=padding
                    )
                )
                cin = cout

        def forward(self, inp):
            flops = torch.tensor(0)
            out = inp
            for layer, k_size in zip(self.layers, layer_kernels):
                cin = out.shape[1]
                out = layer(out)
                # +3 for counting the additions to the flops variable itself
                flops += out.numel() * k_size * k_size * cin + 3

            return out, flops

    model = ConvModel()
    inp = torch.randn(input_shape)
    _, correct_flops = model(inp)

    counted_flops, _, unsupported_ops = count_flops(model, (inp,))
    assert torch.allclose(torch.tensor(counted_flops).long(), correct_flops)
    assert len(unsupported_ops) == 0


@pytest.mark.parametrize("input_shape", ((10, 3), (1, 1), (5, 9), (10, 9, 8)))
def test_flop_count_elementwise(input_shape):
    # Tests that the flop counter counts flops correctly (i.e.
    # when compared to a reference value) for a model involving
    # lots of binary and elementwise operations.

    class ElementwiseModel(torch.nn.Module):
        def forward(self, inp):
            flops = torch.tensor(0)
            out = inp

            out = out * 2
            # +1 here and below for counting the additions to the flop
            # variable itself.
            flops += out.numel() + 1

            out = out.pow(2).exp().log()
            flops += out.numel() * 3 + 1

            out = out.view(-1, 1) @ out.view(1, -1)
            flops += out.numel() + 1

            out = out + out
            flops += out.numel() + 1

            return out, flops

    model = ElementwiseModel()
    inp = torch.randn(input_shape)
    _, correct_flops = model(inp)

    counted_flops, _, unsupported_ops = count_flops(
        model, (inp,), use_single_flop_estimates=True
    )
    assert torch.allclose(torch.tensor(counted_flops).long(), correct_flops)
    assert len(unsupported_ops) == 0


@pytest.mark.parametrize(
    "batch_size, network_channels, compression_channels, img_size", [(1, 8, 16, 128)]
)
def test_flop_count_hyperprior(
    batch_size, network_channels, compression_channels, img_size
):
    # Testing that the flop counter doesn't crash and records all operators
    # on a more complicated model, i.e. the scale hyperprior model.
    model = ScaleHyperprior(
        network_channels=network_channels, compression_channels=compression_channels
    )
    inputs = (torch.randn(batch_size, 3, img_size, img_size),)
    _, _, unsupported_ops = count_flops(model, inputs, use_single_flop_estimates=True)
    assert len(unsupported_ops) == 1


def test_flop_counter_overrides():
    # Testing that one can override the counter's default implementation for
    # an operation with a custom counter function.

    class Addition(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    add_override_value = 1234567.0

    count, _, _ = count_flops(
        Addition(),
        (torch.tensor(5.0), torch.tensor(6.0)),
        counter_overrides={"aten::add": lambda inputs, outputs: add_override_value},
    )

    assert torch.allclose(
        torch.tensor(count).float(), torch.tensor(add_override_value).float()
    )
