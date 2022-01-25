# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.torchscript.framework import TestUtils
from torch_mlir_e2e_test.torchscript.registry import register_test_case
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export

# ==============================================================================

class ReduceSumModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.sum(a)


@register_test_case(module_factory=lambda: ReduceSumModule())
def ReduceSumModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))

# ==============================================================================

class ReduceSumDtypeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float64, True),
    ])
    def forward(self, a):
        return torch.sum(a, dtype=torch.float32)


@register_test_case(module_factory=lambda: ReduceSumDtypeModule())
def ReduceSumDtypeModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5).to(torch.float64))

# ==============================================================================

class ReduceSumDimIntListModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.sum(a, (0, 1))


@register_test_case(module_factory=lambda: ReduceSumDimIntListModule())
def ReduceSumDimIntListModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))

# ==============================================================================

class ReduceSumDimIntListDtypeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float64, True),
    ])
    def forward(self, a):
        return torch.sum(a, (0, 1), dtype=torch.float32)


@register_test_case(module_factory=lambda: ReduceSumDimIntListDtypeModule())
def ReduceSumDimIntListDtypeModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5).to(torch.float64))

# ==============================================================================

class ReduceSumDimIntListKeepDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.sum(a, (1, 2), keepdim=True)


@register_test_case(module_factory=lambda: ReduceSumDimIntListKeepDimModule())
def ReduceSumDimIntListKeepDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))

# ==============================================================================

class ReduceMeanDtypeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float64, True),
    ])
    def forward(self, a):
        return torch.mean(a, dtype=torch.float32)


@register_test_case(module_factory=lambda: ReduceMeanDtypeModule())
def ReduceMeanDtypeModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5).to(torch.float64))

# ==============================================================================

class ReduceMaxAlongDim(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float64, True),
    ])
    def forward(self, a):
        return torch.ops.aten.max(a, 1)[0]


@register_test_case(module_factory=lambda: ReduceMaxAlongDim())
def ReduceMaxAlongDim_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5).to(torch.float64))

# ==============================================================================

class ReduceMaxAlongDimNegative(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float64, True),
    ])
    def forward(self, a):
        return torch.ops.aten.max(a, 1)[0]


@register_test_case(module_factory=lambda: ReduceMaxAlongDimNegative())
def ReduceMaxAlongDimNegative_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5, low=-10, high=10).to(torch.float64))

# ==============================================================================

class ReduceMaxKeepDim(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float64, True),
    ])
    def forward(self, a):
        return torch.ops.aten.max(a, 1, keepdim=True)[1]


@register_test_case(module_factory=lambda: ReduceMaxKeepDim())
def ReduceMaxKeepDim_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5).to(torch.float64))

# ==============================================================================

class ReduceMaxKeepDimReturnBoth(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.ops.aten.max(a, 1, keepdim=True)

@register_test_case(module_factory=lambda: ReduceMaxKeepDimReturnBoth())
def ReduceMaxKeepDimReturnBoth_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5, low=-10, high=-5))

# ==============================================================================

class ReduceMaxAllDims(torch.nn.Module):

  def __init__(self):
    super().__init__()

  @export
  @annotate_args([
      None,
      ([-1, -1, -1], torch.float32, True),
  ])
  def forward(self, a):
    return torch.ops.aten.max(a)

@register_test_case(module_factory=lambda: ReduceMaxAllDims())
def ReduceMaxAllDims_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5, low=-10, high=-5))
