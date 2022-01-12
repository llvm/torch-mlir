# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.torchscript.framework import TestUtils
from torch_mlir_e2e_test.torchscript.registry import register_test_case
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export

# ==============================================================================


class NllLossModule(torch.nn.Module):

  def __init__(self):
    super().__init__()

  @export
  @annotate_args([
      None,
      ([-1, -1], torch.float32, True),
      ([-1], torch.int64, True),
  ])
  # Here the 2nd index is ignored.
  def forward(self, x, y):
    return torch.ops.aten.nll_loss_forward(x,
                                           target=y,
                                           weight=None,
                                           reduction=0,
                                           ignore_index=2)[0]


@register_test_case(module_factory=lambda: NllLossModule())
def NllLossModule_basic(module, tu: TestUtils):
  module.forward(tu.rand(2, 3), torch.tensor([0, 1]))


class NllLossModule_ignore_index_out_of_bounds(torch.nn.Module):

  def __init__(self):
    super().__init__()

  @export
  @annotate_args([
      None,
      ([-1, -1], torch.float32, True),
      ([-1], torch.int64, True),
  ])
  # None of the index is ignored here, since the ignored index is out of bounds.
  def forward(self, x, y):
    return torch.ops.aten.nll_loss_forward(x,
                                           target=y,
                                           weight=None,
                                           reduction=0,
                                           ignore_index=10)[0]


@register_test_case(module_factory=lambda: NllLossModule_ignore_index_out_of_bounds())
def NllLossModule_ignore_index(module, tu: TestUtils):
  module.forward(tu.rand(2, 3), torch.tensor([0, 1]))
