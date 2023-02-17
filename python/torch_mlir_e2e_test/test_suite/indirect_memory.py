# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================

class ScatterReduceFloatModule(torch.nn.Module):
    include_self: bool
    reduce_type: str

    def __init__(self, reduce_type: str, include_self: bool):
        super().__init__()
        self.include_self = include_self
        self.reduce_type = reduce_type

    @export
    @annotate_args([
        None,
        ([10, 8, 6], torch.float32, True),
        ([2, 4, 3], torch.int64, True),
        ([5, 8, 6], torch.float32, True),
    ])
    def forward(self, input, index, src):
        return torch.ops.aten.scatter_reduce(input, 0, index, src, self.reduce_type, include_self=self.include_self)


@register_test_case(
    module_factory=lambda: ScatterReduceFloatModule("sum", False))
def ScatterReduceFloatSumModule(module, tu: TestUtils):
    module.forward(tu.rand(10, 8, 6), tu.randint(2, 4, 3, high=4),
                   tu.rand(5, 8, 6))
@register_test_case(
    module_factory=lambda: ScatterReduceFloatModule("sum", True))
def ScatterReduceFloatSumModuleIncludeSelf(module, tu: TestUtils):
    module.forward(tu.rand(10, 8, 6), tu.randint(2, 4, 3, high=4),
                   tu.rand(5, 8, 6))
@register_test_case(
    module_factory=lambda: ScatterReduceFloatModule("prod", False))
def ScatterReduceFloatProdModule(module, tu: TestUtils):
    module.forward(tu.rand(10, 8, 6), tu.randint(2, 4, 3, high=4),
                   tu.rand(5, 8, 6))
@register_test_case(
    module_factory=lambda: ScatterReduceFloatModule("prod", True))
def ScatterReduceFloatProdModuleIncludeSelf(module, tu: TestUtils):
    module.forward(tu.rand(10, 8, 6), tu.randint(2, 4, 3, high=4),
                   tu.rand(5, 8, 6))
@register_test_case(
    module_factory=lambda: ScatterReduceFloatModule("amax", False))
def ScatterReduceFloatMaxModule(module, tu: TestUtils):
    module.forward(tu.rand(10, 8, 6), tu.randint(2, 4, 3, high=4),
                   tu.rand(5, 8, 6))
@register_test_case(
    module_factory=lambda: ScatterReduceFloatModule("amax", True))
def ScatterReduceFloatMaxModuleIncludeSelf(module, tu: TestUtils):
    module.forward(tu.rand(10, 8, 6), tu.randint(2, 4, 3, high=4),
                   tu.rand(5, 8, 6))
@register_test_case(
    module_factory=lambda: ScatterReduceFloatModule("amin", False))
def ScatterReduceFloatMinModule(module, tu: TestUtils):
    module.forward(tu.rand(10, 8, 6), tu.randint(2, 4, 3, high=4),
                   tu.rand(5, 8, 6))
@register_test_case(
    module_factory=lambda: ScatterReduceFloatModule("amin", True))
def ScatterReduceFloatMinModuleIncludeSelf(module, tu: TestUtils):
    module.forward(tu.rand(10, 8, 6), tu.randint(2, 4, 3, high=4),
                   tu.rand(5, 8, 6))
@register_test_case(
    module_factory=lambda: ScatterReduceFloatModule("mean", False))
def ScatterReduceFloatMeanModule(module, tu: TestUtils):
    module.forward(tu.rand(10, 8, 6), tu.randint(2, 4, 3, high=4),
                   tu.rand(5, 8, 6))
@register_test_case(
    module_factory=lambda: ScatterReduceFloatModule("mean", True))
def ScatterReduceFloatMeanModuleIncludeSelf(module, tu: TestUtils):
    module.forward(tu.rand(10, 8, 6), tu.randint(2, 4, 3, high=4),
                   tu.rand(5, 8, 6))

# ==============================================================================

class ScatterReduceIntModule(torch.nn.Module):
    include_self: bool
    reduce_type: str

    def __init__(self, reduce_type: str, include_self: bool):
        super().__init__()
        self.include_self = include_self
        self.reduce_type = reduce_type

    @export
    @annotate_args([
        None,
        ([10, 8, 6], torch.int32, True),
        ([2, 4, 3], torch.int64, True),
        ([5, 8, 6], torch.int32, True),
    ])
    def forward(self, input, index, src):
        return torch.ops.aten.scatter_reduce(input, 0, index, src, self.reduce_type, include_self=self.include_self)


@register_test_case(
    module_factory=lambda: ScatterReduceIntModule("sum", False))
def ScatterReduceIntSumModule(module, tu: TestUtils):
    module.forward(tu.randint(10, 8, 6, dtype=torch.int32, high=10), tu.randint(2, 4, 3, high=4),
                   tu.randint(5, 8, 6, dtype=torch.int32, high=10))
@register_test_case(
    module_factory=lambda: ScatterReduceIntModule("sum", True))
def ScatterReduceIntSumModuleIncludeSelf(module, tu: TestUtils):
    module.forward(tu.randint(10, 8, 6, dtype=torch.int32, high=10), tu.randint(2, 4, 3, high=4),
                   tu.randint(5, 8, 6, dtype=torch.int32, high=10))
@register_test_case(
    module_factory=lambda: ScatterReduceIntModule("prod", False))
def ScatterReduceIntProdModule(module, tu: TestUtils):
    module.forward(tu.randint(10, 8, 6, dtype=torch.int32, high=10), tu.randint(2, 4, 3, high=4),
                   tu.randint(5, 8, 6, dtype=torch.int32, high=10))
@register_test_case(
    module_factory=lambda: ScatterReduceIntModule("prod", True))
def ScatterReduceIntProdModuleIncludeSelf(module, tu: TestUtils):
    module.forward(tu.randint(10, 8, 6, dtype=torch.int32, high=10), tu.randint(2, 4, 3, high=4),
                   tu.randint(5, 8, 6, dtype=torch.int32, high=10))
@register_test_case(
    module_factory=lambda: ScatterReduceIntModule("amax", False))
def ScatterReduceIntMaxModule(module, tu: TestUtils):
    module.forward(tu.randint(10, 8, 6, dtype=torch.int32, high=10), tu.randint(2, 4, 3, high=4),
                   tu.randint(5, 8, 6, dtype=torch.int32, high=10))
@register_test_case(
    module_factory=lambda: ScatterReduceIntModule("amax", True))
def ScatterReduceIntMaxModuleIncludeSelf(module, tu: TestUtils):
    module.forward(tu.randint(10, 8, 6, dtype=torch.int32, high=10), tu.randint(2, 4, 3, high=4),
                   tu.randint(5, 8, 6, dtype=torch.int32, high=10))
@register_test_case(
    module_factory=lambda: ScatterReduceIntModule("amin", False))
def ScatterReduceIntMinModule(module, tu: TestUtils):
    module.forward(tu.randint(10, 8, 6, dtype=torch.int32, high=10), tu.randint(2, 4, 3, high=4),
                   tu.randint(5, 8, 6, dtype=torch.int32, high=10))
@register_test_case(
    module_factory=lambda: ScatterReduceIntModule("amin", True))
def ScatterReduceIntMinModuleIncludeSelf(module, tu: TestUtils):
    module.forward(tu.randint(10, 8, 6, dtype=torch.int32, high=10), tu.randint(2, 4, 3, high=4),
                   tu.randint(5, 8, 6, dtype=torch.int32, high=10))
@register_test_case(
    module_factory=lambda: ScatterReduceIntModule("mean", False))
def ScatterReduceIntMeanModule(module, tu: TestUtils):
    module.forward(tu.randint(10, 8, 6, dtype=torch.int32, high=10), tu.randint(2, 4, 3, high=4),
                   tu.randint(5, 8, 6, dtype=torch.int32, high=10))
@register_test_case(
    module_factory=lambda: ScatterReduceIntModule("mean", True))
def ScatterReduceIntMeanModuleIncludeSelf(module, tu: TestUtils):
    module.forward(tu.randint(10, 8, 6, dtype=torch.int32, high=10), tu.randint(2, 4, 3, high=4),
                   tu.randint(5, 8, 6, dtype=torch.int32, high=10))

# ==============================================================================
