#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch

from torch_mlir_e2e_test.torchscript.framework import TestUtils
from torch_mlir_e2e_test.torchscript.registry import register_test_case
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export

# ==============================================================================

class ViewExpandModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([6, 4], torch.float32, True),
    ])

    def forward(self, a):
        return a.view(2, 3, 4)

@register_test_case(module_factory=lambda: ViewExpandModule())
def ViewExpandModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6, 4))

# ==============================================================================

class ViewDynamicExpandModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, 30, 384], torch.float32, True),
    ])

    def forward(self, a):
        return a.view(2, 4, 5, 6, 12, 32)

@register_test_case(module_factory=lambda: ViewDynamicExpandModule())
def ViewDynamicExpandModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4, 30, 384))

