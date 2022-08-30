# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================

# Custom operators must be registered with PyTorch before being used.
# This is part of the test.
# Note that once this library has been loaded, the side effects mutate
# the PyTorch op registry permanently.
import torch_mlir._torch_mlir_custom_op_example

class CustomOpExampleModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.ops._torch_mlir_custom_op_example.identity(a)


@register_test_case(module_factory=lambda: CustomOpExampleModule())
def CustomOpExampleModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))
