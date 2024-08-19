# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export


# ==============================================================================
class TimeOutModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1, -1], torch.int64, True)])
    def forward(self, x):
        x_val = x.size(0)  # this is going to be 2
        sum = 100
        while x_val < sum:  # sum will always > 2
            sum += 1
        return sum


@register_test_case(module_factory=lambda: TimeOutModule(), timeout=10)
def TimeOutModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(6, 8, high=10))
