# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================

class ScatterReduceModule(torch.nn.Module):
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
    module_factory=lambda: ScatterReduceModule("sum", False))
def ScatterReduceSumModule(module, tu: TestUtils):
    module.forward(tu.rand(10, 8, 6), tu.randint(2, 4, 3, high=4),
                   tu.rand(5, 8, 6))
@register_test_case(
    module_factory=lambda: ScatterReduceModule("sum", True))
def ScatterReduceSumModuleIncludeSelf(module, tu: TestUtils):
    module.forward(tu.rand(10, 8, 6), tu.randint(2, 4, 3, high=4),
                   tu.rand(5, 8, 6))
@register_test_case(
    module_factory=lambda: ScatterReduceModule("prod", False))
def ScatterReduceProdModule(module, tu: TestUtils):
    module.forward(tu.rand(10, 8, 6), tu.randint(2, 4, 3, high=4),
                   tu.rand(5, 8, 6))
@register_test_case(
    module_factory=lambda: ScatterReduceModule("prod", True))
def ScatterReduceProdModuleIncludeSelf(module, tu: TestUtils):
    module.forward(tu.rand(10, 8, 6), tu.randint(2, 4, 3, high=4),
                   tu.rand(5, 8, 6))
@register_test_case(
    module_factory=lambda: ScatterReduceModule("amax", False))
def ScatterReduceMaxModule(module, tu: TestUtils):
    module.forward(tu.rand(10, 8, 6), tu.randint(2, 4, 3, high=4),
                   tu.rand(5, 8, 6))
@register_test_case(
    module_factory=lambda: ScatterReduceModule("amax", True))
def ScatterReduceMaxModuleIncludeSelf(module, tu: TestUtils):
    module.forward(tu.rand(10, 8, 6), tu.randint(2, 4, 3, high=4),
                   tu.rand(5, 8, 6))
@register_test_case(
    module_factory=lambda: ScatterReduceModule("amin", False))
def ScatterReduceMinModule(module, tu: TestUtils):
    module.forward(tu.rand(10, 8, 6), tu.randint(2, 4, 3, high=4),
                   tu.rand(5, 8, 6))
@register_test_case(
    module_factory=lambda: ScatterReduceModule("amin", True))
def ScatterReduceMinModuleIncludeSelf(module, tu: TestUtils):
    module.forward(tu.rand(10, 8, 6), tu.randint(2, 4, 3, high=4),
                   tu.rand(5, 8, 6))

# ==============================================================================
