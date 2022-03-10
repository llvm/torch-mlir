# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.torchscript.framework import TestUtils
from torch_mlir_e2e_test.torchscript.registry import register_test_case
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export

# ==============================================================================

class IndexPutImplOneDimFloatNonAccumulateModule(torch.nn.Module):

  def __init__(self):
    super().__init__()

  @export
  @annotate_args([
      None,
      ([-1], torch.float32, True),
      ([-1], torch.int64, True),
      ([-1], torch.float32, True),
  ])
  def forward(self, input, index, value):
    return torch.ops.aten._index_put_impl_(input, (index,), value,
                                          accumulate=False,
                                          unsafe=False)


@register_test_case(module_factory=lambda: IndexPutImplOneDimFloatNonAccumulateModule())
def IndexPutImplOneDimFloatNonAccumulateModule_basic(module, tu: TestUtils):
  module.forward(tu.rand(100), torch.randint(100, (250,)),
                 tu.rand(250))


class IndexPutImplOneDimIntNonAccumulateModule(torch.nn.Module):

  def __init__(self):
    super().__init__()

  @export
  @annotate_args([
      None,
      ([-1], torch.int64, True),
      ([-1], torch.int64, True),
      ([-1], torch.int64, True),
  ])
  def forward(self, input, index, value):
    return torch.ops.aten._index_put_impl_(input, (index,), value,
                                          accumulate=False,
                                          unsafe=False)


@register_test_case(module_factory=lambda: IndexPutImplOneDimIntNonAccumulateModule())
def IndexPutImplOneDimIntNonAccumulateModule_basic(module, tu: TestUtils):
  module.forward(torch.randint(1000, (200,)), torch.randint(100, (300,)),
                 torch.randint(10000, (300,)))


class IndexPutImplOneDimFloatAccumulateModule(torch.nn.Module):

  def __init__(self):
    super().__init__()

  @export
  @annotate_args([
      None,
      ([-1], torch.float32, True),
      ([-1], torch.int64, True),
      ([-1], torch.float32, True),
  ])
  def forward(self, input, index, value):
    # Since the input is updated in-place, we pass input.clone() in place
    # of input to avoid wrong results.
    return torch.ops.aten._index_put_impl_(input.clone(), (index,), value,
                                          accumulate=True,
                                          unsafe=False)


@register_test_case(module_factory=lambda: IndexPutImplOneDimFloatAccumulateModule())
def IndexPutImplOneDimFloatAccumulateModule_basic(module, tu: TestUtils):
  module.forward(tu.rand(1000), torch.randint(10, (500,)),
                 tu.rand(500))


class IndexPutImplOneDimIntAccumulateModule(torch.nn.Module):

  def __init__(self):
    super().__init__()

  @export
  @annotate_args([
      None,
      ([-1], torch.int64, True),
      ([-1], torch.int64, True),
      ([-1], torch.int64, True),
  ])
  def forward(self, input, index, value):
    # Since the input is updated in-place, we pass input.clone() in place
    # of input to avoid wrong results.
    return torch.ops.aten._index_put_impl_(input.clone(), (index,), value,
                                          accumulate=True,
                                          unsafe=False)


@register_test_case(module_factory=lambda: IndexPutImplOneDimIntAccumulateModule())
def IndexPutImplOneDimIntAccumulateModule_basic(module, tu: TestUtils):
  module.forward(torch.randint(100, (10,)), torch.randint(10, (10,)),
                 torch.randint(1000, (10,)))

# ==============================================================================

class IndexPutOneDimFloatNonAccumulateModule(torch.nn.Module):

  def __init__(self):
    super().__init__()

  @export
  @annotate_args([
      None,
      ([-1], torch.float32, True),
      ([-1], torch.int64, True),
      ([-1], torch.float32, True),
  ])
  def forward(self, input, index, value):
    return torch.ops.aten.index_put(input, (index,), value, accumulate=False)


@register_test_case(module_factory=lambda: IndexPutOneDimFloatNonAccumulateModule())
def IndexPutOneDimFloatNonAccumulateModule_basic(module, tu: TestUtils):
  module.forward(tu.rand(100), torch.randint(100, (250,)),
                 tu.rand(250))


class IndexPutOneDimIntNonAccumulateModule(torch.nn.Module):

  def __init__(self):
    super().__init__()

  @export
  @annotate_args([
      None,
      ([-1], torch.int64, True),
      ([-1], torch.int64, True),
      ([-1], torch.int64, True),
  ])
  def forward(self, input, index, value):
    return torch.ops.aten.index_put(input, (index,), value, accumulate=False)


@register_test_case(module_factory=lambda: IndexPutOneDimIntNonAccumulateModule())
def IndexPutOneDimIntNonAccumulateModule_basic(module, tu: TestUtils):
  module.forward(torch.randint(1000, (200,)), torch.randint(100, (300,)),
                 torch.randint(10000, (300,)))


class IndexPutOneDimFloatAccumulateModule(torch.nn.Module):

  def __init__(self):
    super().__init__()

  @export
  @annotate_args([
      None,
      ([-1], torch.float32, True),
      ([-1], torch.int64, True),
      ([-1], torch.float32, True),
  ])
  def forward(self, input, index, value):
    return torch.ops.aten.index_put(input, (index,), value, accumulate=True)


@register_test_case(module_factory=lambda: IndexPutOneDimFloatAccumulateModule())
def IndexPutOneDimFloatAccumulateModule_basic(module, tu: TestUtils):
  module.forward(tu.rand(1000), torch.randint(10, (500,)),
                 tu.rand(500))


class IndexPutOneDimIntAccumulateModule(torch.nn.Module):

  def __init__(self):
    super().__init__()

  @export
  @annotate_args([
      None,
      ([-1], torch.int64, True),
      ([-1], torch.int64, True),
      ([-1], torch.int64, True),
  ])
  def forward(self, input, index, value):
    return torch.ops.aten.index_put(input, (index,), value, accumulate=True)


@register_test_case(module_factory=lambda: IndexPutOneDimIntAccumulateModule())
def IndexPutOneDimIntAccumulateModule_basic(module, tu: TestUtils):
  module.forward(torch.randint(100, (10,)), torch.randint(10, (10,)),
                 torch.randint(1000, (10,)))
