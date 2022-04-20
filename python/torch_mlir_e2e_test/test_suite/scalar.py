# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.torchscript.framework import TestUtils
from torch_mlir_e2e_test.torchscript.registry import register_test_case
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export

# ==============================================================================

class AddIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([], torch.int64, True),
        ([], torch.int64, True),
    ])
    def forward(self, lhs, rhs):
        return int(lhs)+int(rhs)


@register_test_case(module_factory=lambda: AddIntModule())
def AddIntModule_basic(module, tu: TestUtils):
    module.forward(torch.randint(-100, 100,()), torch.randint(-100, 100,()))

# ==============================================================================

class SubIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([], torch.int64, True),
        ([], torch.int64, True),
    ])
    def forward(self, lhs, rhs):
        return int(lhs)-int(rhs)


@register_test_case(module_factory=lambda: SubIntModule())
def SubIntModule_basic(module, tu: TestUtils):
    module.forward(torch.randint(-100, 100,()), torch.randint(-100, 100,()))

# ==============================================================================

class SubFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([], torch.float64, True),
        ([], torch.float64, True),
    ])
    def forward(self, lhs, rhs):
        return float(lhs)-float(rhs)


@register_test_case(module_factory=lambda: SubFloatModule())
def SubFloatModule_basic(module, tu: TestUtils):
    module.forward(torch.rand(()).double(), torch.rand(()).double())

# ==============================================================================

class MulIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([], torch.int64, True),
        ([], torch.int64, True),
    ])
    def forward(self, lhs, rhs):
        return int(lhs)*int(rhs)


@register_test_case(module_factory=lambda: MulIntModule())
def MulIntModule_basic(module, tu: TestUtils):
    module.forward(torch.randint(-100, 100,()), torch.randint(-100, 100,()))

# ==============================================================================
