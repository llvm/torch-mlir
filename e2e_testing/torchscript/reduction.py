# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.torchscript.framework import TestUtils
from torch_mlir_e2e_test.torchscript.registry import register_test_case
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export

# ==============================================================================

class ReduceSumModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.sum(a)


@register_test_case(module_factory=lambda: ReduceSumModule())
def ReduceSumModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))

# ==============================================================================

class ReduceSumDimIntListModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.sum(a, (0, 1))


@register_test_case(module_factory=lambda: ReduceSumDimIntListModule())
def ReduceSumDimIntListModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))

# ==============================================================================

class ReduceSumDimIntListKeepDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.sum(a, (1, 2), keepdim=True)


@register_test_case(module_factory=lambda: ReduceSumDimIntListKeepDimModule())
def ReduceSumDimIntListKeepDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))
