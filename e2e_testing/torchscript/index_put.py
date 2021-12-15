# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.torchscript.framework import TestUtils
from torch_mlir_e2e_test.torchscript.registry import register_test_case
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export

# ==============================================================================


# Updating a 2d tensor from 0d indices and value.
class IndexPutZeroModule(torch.nn.Module):

  def __init__(self):
    super().__init__()

  @export
  @annotate_args([
      None,
      ([-1, -1], torch.float32, True),
      ([], torch.int64, True),
      ([], torch.int64, True),
      ([], torch.float32, True),
  ])
  def forward(self, input, index1, index2, value):
    return torch.ops.aten.index_put(input, (index1, index2), value)


@register_test_case(module_factory=lambda: IndexPutZeroModule())
def IndexPutZeroModule_basic(module, tu: TestUtils):
  module.forward(tu.rand(2, 2), torch.tensor(1, dtype=torch.int64),
                 torch.tensor(1, dtype=torch.int64), tu.rand())


# Updating a 3d tensor from 2d indices and value.
class IndexPut3dModule(torch.nn.Module):

  def __init__(self):
    super().__init__()

  @export
  @annotate_args([
      None,
      ([-1, -1, -1], torch.float32, True),
      ([-1, -1], torch.int64, True),
      ([-1, -1], torch.int64, True),
      ([-1, -1], torch.int64, True),
      ([-1, -1], torch.float32, True),
  ])
  def forward(self, input, index1, index2, index3, value):
    return torch.ops.aten.index_put(input, (index1, index2, index3), value)


@register_test_case(module_factory=lambda: IndexPut3dModule())
def IndexPut3dModule_basic(module, tu: TestUtils):
  module.forward(tu.rand(5, 5, 5), torch.randint(5, (2, 2)),
                 torch.randint(1, 4, (2, 2)), torch.randint(0, 4, (2, 2)),
                 tu.rand(2, 2))

# Updating a 3d tensor from arbitrary equivalent indices and value.
class IndexPutEquivalentIndicesModule(torch.nn.Module):

  def __init__(self):
    super().__init__()

  @export
  @annotate_args([
      None,
      ([-1, -1, -1], torch.float32, True),
      ([-1, -1], torch.int64, True),
      ([-1], torch.int64, True),
      ([-1, -1, -1], torch.int64, True),
      ([-1, -1], torch.float32, True),
  ])
  def forward(self, input, index1, index2, index3, value):
    return torch.ops.aten.index_put(input, (index1, index2, index3), value)


@register_test_case(module_factory=lambda: IndexPutEquivalentIndicesModule())
def IndexPutEquivalentIndicesModule_basic(module, tu: TestUtils):
  module.forward(tu.rand(5, 5, 5), torch.randint(5, (1, 2)),
                 torch.randint(1, 4, (2,)), torch.randint(0, 4, (1, 1, 2)),
                 tu.rand(1, 2))
