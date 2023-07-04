# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import functorch
import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================

class ArangeDerefineModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        end = torch.ops.prim.NumToTensor(x.shape[1])
        return torch.arange(0, end, 1)

@register_test_case(module_factory=lambda: ArangeDerefineModule())
def ArangeDerefineModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 4))

# ==============================================================================
