# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.torchscript.framework import TestUtils
from torch_mlir_e2e_test.torchscript.registry import register_test_case
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export

# ==============================================================================

class ReduceSumFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.sum(a)


@register_test_case(module_factory=lambda: ReduceSumFloatModule())
def ReduceSumFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))

# ==============================================================================

class ReduceSumDtypeFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float64, True),
    ])
    def forward(self, a):
        return torch.sum(a, dtype=torch.float32)


@register_test_case(module_factory=lambda: ReduceSumDtypeFloatModule())
def ReduceSumDtypeFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5).to(torch.float64))

# ==============================================================================

class ReduceSumDimIntListFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.sum(a, (0, 1))


@register_test_case(module_factory=lambda: ReduceSumDimIntListFloatModule())
def ReduceSumDimIntListFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))

# ==============================================================================

class ReduceSumDimIntListDtypeFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float64, True),
    ])
    def forward(self, a):
        return torch.sum(a, (0, 1), dtype=torch.float32)


@register_test_case(module_factory=lambda: ReduceSumDimIntListDtypeFloatModule())
def ReduceSumDimIntListDtypeFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5).to(torch.float64))

# ==============================================================================

class ReduceSumDimIntListKeepDimFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.sum(a, (1, 2), keepdim=True)


@register_test_case(module_factory=lambda: ReduceSumDimIntListKeepDimFloatModule())
def ReduceSumDimIntListKeepDimFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))

# ==============================================================================

class ReduceSumUnsignedIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.int64, True),
    ])
    def forward(self, a):
        return torch.sum(a)


@register_test_case(module_factory=lambda: ReduceSumUnsignedIntModule())
def ReduceSumUnsignedIntModule_basic(module, tu: TestUtils):
    module.forward(torch.randint(0, 100, (3, 4, 5)))

# ==============================================================================

class ReduceSumSignedIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.int64, True),
    ])
    def forward(self, a):
        return torch.sum(a)


@register_test_case(module_factory=lambda: ReduceSumSignedIntModule())
def ReduceSumSignedIntModule_basic(module, tu: TestUtils):
    module.forward(torch.randint(-100, 100, (3, 4, 5)))

# ==============================================================================

class ReduceSumDtypeIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.int32, True),
    ])
    def forward(self, a):
        return torch.sum(a, dtype=torch.int64)


@register_test_case(module_factory=lambda: ReduceSumDtypeIntModule())
def ReduceSumDtypeIntModule_basic(module, tu: TestUtils):
    module.forward(torch.randint(100, (3, 4, 5)).to(torch.int32))

# ==============================================================================

class ReduceSumDimIntListIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.int64, True),
    ])
    def forward(self, a):
        return torch.sum(a, (0, 1))


@register_test_case(module_factory=lambda: ReduceSumDimIntListIntModule())
def ReduceSumDimIntListIntModule_basic(module, tu: TestUtils):
    module.forward(torch.randint(100, (3, 4, 5)))

# ==============================================================================

class ReduceSumDimIntListDtypeIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.int32, True),
    ])
    def forward(self, a):
        return torch.sum(a, (0, 1), dtype=torch.int64)


@register_test_case(module_factory=lambda: ReduceSumDimIntListDtypeIntModule())
def ReduceSumDimIntListDtypeIntModule_basic(module, tu: TestUtils):
    module.forward(torch.randint(100, (3, 4, 5)).to(torch.int32))

# ==============================================================================

class ReduceSumDimIntListKeepDimIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.int64, True),
    ])
    def forward(self, a):
        return torch.sum(a, (1, 2), keepdim=True)


@register_test_case(module_factory=lambda: ReduceSumDimIntListKeepDimIntModule())
def ReduceSumDimIntListKeepDimIntModule_basic(module, tu: TestUtils):
    module.forward(torch.randint(100, (3, 4, 5)))

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

# ==============================================================================

class ReduceMaxNegativeDim(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.ops.aten.max(a, -1, keepdim=True)

@register_test_case(module_factory=lambda: ReduceMaxNegativeDim())
def ReduceMaxNegativeDim_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))

# ==============================================================================

class ReduceMaxFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.ops.aten.max(a)

@register_test_case(module_factory=lambda: ReduceMaxFloatModule())
def ReduceMaxFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))

# ==============================================================================

class ReduceMaxSignedIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.int64, True),
    ])
    def forward(self, a):
        return torch.ops.aten.max(a)

@register_test_case(module_factory=lambda: ReduceMaxSignedIntModule())
def ReduceMaxSignedIntModule_basic(module, tu: TestUtils):
    module.forward(torch.randint(-100, 100, (3, 4, 5)))

# ==============================================================================

class ReduceMaxUnsignedIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.int64, True),
    ])
    def forward(self, a):
        return torch.ops.aten.max(a)

@register_test_case(module_factory=lambda: ReduceMaxUnsignedIntModule())
def ReduceMaxUnsignedIntModule_basic(module, tu: TestUtils):
    module.forward(torch.randint(100, (3, 4, 5)))
