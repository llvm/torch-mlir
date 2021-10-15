# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.torchscript.framework import TestUtils
from torch_mlir_e2e_test.torchscript.registry import register_test_case
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export

# ==============================================================================

class TypeConversionF32ToF64Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True)
    ])
    def forward(self, x):
        return x.to(torch.float64)

@register_test_case(module_factory=lambda: TypeConversionF32ToF64Module())
def TypeConversionF32ToF64Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5))


class TypeConversionF64ToF32Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float64, True)
    ])
    def forward(self, x):
        return x.to(torch.float32)

@register_test_case(module_factory=lambda: TypeConversionF64ToF32Module())
def TypeConversionF64ToF32Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5).type(torch.float64))

class TypeConversionI32ToI64Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True)
    ])
    def forward(self, x):
        return x.to(torch.int64)

@register_test_case(module_factory=lambda: TypeConversionI32ToI64Module())
def TypeConversionI32ToI64Module_basic(module, tu: TestUtils):
    module.forward(torch.randint(5, [2, 3]).type(torch.int32))

class TypeConversionI64ToI32Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True)
    ])
    def forward(self, x):
        return x.to(torch.int32)

@register_test_case(module_factory=lambda: TypeConversionI64ToI32Module())
def TypeConversionI64ToI32Module_basic(module, tu: TestUtils):
    module.forward(torch.randint(5, [2, 3]))
