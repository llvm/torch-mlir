#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from torch_mlir_torchscript.e2e_test.framework import TestUtils
from torch_mlir_torchscript.e2e_test.registry import register_test_case
from torch_mlir_torchscript.annotations import annotate_args, export

# ==============================================================================

class Conv2dNoPaddingModule(torch.nn.Module):
  def __init__(self):
    super().__init__()
    torch.manual_seed(0)
    self.conv = torch.nn.Conv2d(2, 10, 3, bias = False)
    self.train(False)

  @export
  @annotate_args([
      None,
      ([-1, -1, -1, -1], torch.float32, True),
  ])
  def forward(self, x):
    return self.conv(x)

@register_test_case(module_factory=lambda: Conv2dNoPaddingModule())
def Conv2dNoPaddingModule_basic(module, tu: TestUtils):
    t = tu.rand(5, 2, 10, 20)
    module.forward(t)

class Conv2dWithPaddingModule(torch.nn.Module):
  def __init__(self):
    super().__init__()
    torch.manual_seed(0)
    self.conv = torch.nn.Conv2d(2, 10, 3, bias = False, padding = 3)
    self.train(False)

  @export
  @annotate_args([
      None,
      ([-1, -1, -1, -1], torch.float32, True),
  ])
  def forward(self, x):
    return self.conv(x)

@register_test_case(module_factory=lambda: Conv2dWithPaddingModule())
def Conv2dWithPaddingModule_basic(module, tu: TestUtils):
    t = tu.rand(5, 2, 10, 20)
    module.forward(t)
