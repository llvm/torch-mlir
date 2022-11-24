# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================


class TestMultipleTensorReturn(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float64, True),
        ([-1, -1], torch.int32, True),
        ([-1, -1], torch.int64, True),
        ([-1, -1], torch.bool, True),
    ])
    def forward(self, a, b, c, d, e):
        return a, b, c, d, e


@register_test_case(module_factory=lambda: TestMultipleTensorReturn())
def TestMultipleTensorReturn_basic(module, tu: TestUtils):
    module.forward(
        tu.rand(3, 4).to(torch.float32),
        tu.rand(2, 3).to(torch.float64),
        tu.rand(2, 3).to(torch.int32),
        tu.rand(2, 3).to(torch.int64),
        tu.rand(2, 3).to(torch.bool))


class TestMultipleTensorAndPrimitiveTypesReturn(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
        ([-1, -1], torch.float64, True),
        ([-1, -1], torch.bool, True),
    ])
    def forward(self, a, b, c):
        d = 1
        e = 2.3
        return a, b, c, d, e


@register_test_case(
    module_factory=lambda: TestMultipleTensorAndPrimitiveTypesReturn())
def TestMultipleTensorAndPrimitiveTypesReturn_basic(module, tu: TestUtils):
    module.forward(
        tu.rand(3, 4).to(torch.int32),
        tu.rand(2, 3).to(torch.float64),
        tu.rand(2, 3).to(torch.bool))


# ==============================================================================
