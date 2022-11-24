# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from numpy import int64
import torch
import random

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================

class TorchPrimLoopForLikeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True)
    ])
    def forward(self, x):
        x_val = x.size(0)
        sum = 0
        for i in range(x_val):
            sum += i
        return sum
        

@register_test_case(module_factory=lambda: TorchPrimLoopForLikeModule())
def TorchPrimLoopForLikeModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(6, 8, high=10))

# ==============================================================================
class TorchPrimLoopWhileLikeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True)
    ])
    def forward(self, x):
        x_val = x.size(0)
        sum = 0
        while(x_val > sum):
            sum += 1
        return sum
        

@register_test_case(module_factory=lambda: TorchPrimLoopWhileLikeModule())
def TorchPrimLoopWhileLikeModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(6, 8, high=10))
