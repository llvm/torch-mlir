# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.torchscript.framework import TestUtils
from torch_mlir_e2e_test.torchscript.registry import register_test_case
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export

# ==============================================================================

# Custom operators must be registered with PyTorch before being used.
# This is part of the test.
# Note that once this library has been loaded, the side effects mutate
# the PyTorch op registry permanently.
try:
  import torch_mlir._torch_mlir_custom_op_example
except ImportError:
  # Delay import failure. This allows us to xfail the CustomOp tests at the
  # cost of slightly more complicated error messages later.
  pass

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

