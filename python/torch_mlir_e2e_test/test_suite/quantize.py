# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================


class QuantizeInt8(torch.nn.Module):
    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        y = torch.int_repr(torch.quantize_per_tensor(x, 1.0, 0, torch.qint8))
        y2 = torch.int_repr(torch.quantize_per_tensor(x, 2.0, -100, torch.qint8))
        return y, y2

@register_test_case(module_factory=lambda: QuantizeInt8())
def Quantize_int8(module, tu: TestUtils):
    module.forward(tu.rand(3, 5, low=-10, high=10))
    module.forward(torch.FloatTensor([[-129, -128, -127], [126, 127, 128]]))
    module.forward(torch.FloatTensor([[-1.5, -0.5], [0.5, 1.5]]))

class QuantizeUInt8(torch.nn.Module):
    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        y = torch.int_repr(torch.quantize_per_tensor(x, 1.0, 0, torch.quint8))
        y2 = torch.int_repr(torch.quantize_per_tensor(x, 2.0, 3, torch.quint8))
        return y, y2


@register_test_case(module_factory=lambda: QuantizeUInt8())
def Quantize_uint8(module, tu: TestUtils):
    module.forward(tu.rand(3, 5, low=-10, high=10))
    module.forward(torch.FloatTensor([[-1, 0, 1], [254, 255, 256]]))
    module.forward(torch.FloatTensor([[-1.5, -0.5], [0.5, 1.5]]))
