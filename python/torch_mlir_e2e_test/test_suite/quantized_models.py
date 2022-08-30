# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch
from torch import nn

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================


class QuantizedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        torch.random.manual_seed(0)
        self.layers = nn.Sequential(
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
        )
        self.quantize = torch.quantization.QuantStub()
        self.dequantize = torch.quantization.DeQuantStub()

    @export
    @export
    @annotate_args([
        None,
        ([1, 16], torch.float32, True),
    ])
    def forward(self, x):
        x = self.quantize(x)
        x = self.layers(x)
        x = self.dequantize(x)
        return x


def get_mlp_input():
    return 2 * torch.rand((1, 16)) - 1


def get_quantized_mlp():
    model = QuantizedMLP()
    model.eval()
    model.qconfig = torch.quantization.default_qconfig
    torch.quantization.prepare(model, inplace=True)
    torch.manual_seed(0)
    for _ in range(32):
        model(get_mlp_input())
    torch.quantization.convert(model, inplace=True)
    return model


@register_test_case(module_factory=get_quantized_mlp)
def QuantizedMLP_basic(module, tu: TestUtils):
    module.forward(get_mlp_input())
