# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

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

class ReduceSumElementTypeBoolModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.bool, True),
    ])
    def forward(self, a):
        return torch.sum(a)


@register_test_case(module_factory=lambda: ReduceSumElementTypeBoolModule())
def ReduceSumElementTypeBoolModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, 5, high=2).to(torch.bool))

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

class ReduceSumDimIntListKeepDimNegativeDimStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([1, 12, 7, 7], torch.float32, True),
    ])
    def forward(self, a):
        return torch.sum(a, dim=(-1), keepdim=True)


@register_test_case(module_factory=lambda: ReduceSumDimIntListKeepDimNegativeDimStaticModule())
def ReduceSumDimIntListKeepDimNegativeDimStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 12, 7, 7))

# ==============================================================================

class ReduceSumDimIntListEmptyDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.sum(a, dim=[])


@register_test_case(module_factory=lambda: ReduceSumDimIntListEmptyDimModule())
def ReduceSumDimIntListEmptyDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))

# ==============================================================================

class ReduceSumDimIntListElementTypeBoolModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.bool, True),
    ])
    def forward(self, a):
        return torch.sum(a, dim=(-1), keepdim=False)


@register_test_case(module_factory=lambda: ReduceSumDimIntListElementTypeBoolModule())
def ReduceSumDimIntListElementTypeBoolModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(1, 128, high=2).to(dtype=torch.bool))

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
    module.forward(tu.randint(3, 4, 5, low=0, high=100))

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
    module.forward(tu.randint(3, 4, 5, low=-100, high=100))

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
    module.forward(tu.randint(3, 4, 5, high=100).to(torch.int32))

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
    module.forward(tu.randint(3, 4, 5, high=100))

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
    module.forward(tu.randint(3, 4, 5, high=100).to(torch.int32))

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
    module.forward(tu.randint(3, 4, 5, high=100))

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
    module.forward(tu.randint(3, 4, 5, low=-100, high=100))

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
    module.forward(tu.randint(3, 4, 5, high=100))

# ==============================================================================

class ReduceAmaxSingleDim(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.ops.aten.amax(a, 1)

@register_test_case(module_factory=lambda: ReduceAmaxSingleDim())
def ReduceAmaxSingleDim_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5, high=100))

# ==============================================================================

class ReduceAmaxMultiDim(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.ops.aten.amax(a, (0, 2))

@register_test_case(module_factory=lambda: ReduceAmaxMultiDim())
def ReduceAmaxMultiDim_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5, high=100))

# ==============================================================================

class ReduceAmaxOutOfOrderDim(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.ops.aten.amax(a, (2, 1, 3))

@register_test_case(module_factory=lambda: ReduceAmaxOutOfOrderDim())
def ReduceAmaxOutOfOrderDim_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5, 6, high=100))

# ==============================================================================

class ReduceAmaxKeepDim(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.ops.aten.amax(a, (0, 2), keepdim=True)

@register_test_case(module_factory=lambda: ReduceAmaxKeepDim())
def ReduceAmaxKeepDim_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5, high=100))

# ==============================================================================

class ReduceL1NormModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.linalg.vector_norm(a, dim=0, ord=1)

@register_test_case(module_factory=lambda: ReduceL1NormModule())
def ReduceL1NormModule_basic(module, tu: TestUtils):
    module.forward(torch.rand(3, 4, 5))

# ==============================================================================

class ReduceL1NormWithDTypeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.linalg.vector_norm(a, dim=0, ord=1, dtype=torch.float64)

@register_test_case(module_factory=lambda: ReduceL1NormWithDTypeModule())
def ReduceL1NormWithDTypeModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5).to(torch.float32))

# ==============================================================================

class ReduceL2NormModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.linalg.vector_norm(a, dim=0)

@register_test_case(module_factory=lambda: ReduceL2NormModule())
def ReduceL2NormModule_basic(module, tu: TestUtils):
    module.forward(torch.rand(3, 4, 5))

# ==============================================================================

class ReduceLN3NormModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.linalg.vector_norm(a, dim=0, ord=-3)

@register_test_case(module_factory=lambda: ReduceLN3NormModule())
def ReduceLN3NormModule_basic(module, tu: TestUtils):
    module.forward(torch.rand(3, 4, 5))

# ==============================================================================

class ReduceL3NormAllDimsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.linalg.vector_norm(a, dim=None, ord=3)

@register_test_case(module_factory=lambda: ReduceL3NormAllDimsModule())
def ReduceL3NormAllDimsModule_basic(module, tu: TestUtils):
    module.forward(torch.rand(3, 4, 5))

# ==============================================================================

class ReduceL3NormKeepDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.linalg.vector_norm(a, keepdim=True, ord=3)

@register_test_case(module_factory=lambda: ReduceL3NormKeepDimModule())
def ReduceL3NormKeepDimModule_basic(module, tu: TestUtils):
    module.forward(torch.rand(3, 4, 5))

# ==============================================================================
class ReduceFrobeniusNormModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    @export 
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])  
    def forward(self, a):
        return torch.ops.aten.frobenius_norm(a, dim=[0, 1], keepdim=False)

@register_test_case(module_factory=lambda: ReduceFrobeniusNormModule())
def ReduceFrobeniusNormModule_basic(module, tu: TestUtils):
    module.forward(torch.rand(3, 4, 5))

# ==============================================================================
class ReduceFrobeniusNormKeepDimModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    @export 
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])  
    def forward(self, a):
        return torch.ops.aten.frobenius_norm(a, dim=[0, 1], keepdim=True)

@register_test_case(module_factory=lambda: ReduceFrobeniusNormKeepDimModule())
def ReduceFrobeniusNormKeepDimModule_basic(module, tu: TestUtils):
    module.forward(torch.rand(3, 4, 5))

# ==============================================================================

class MseLossNoReductionModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1 , -1], torch.float32, True),
        ([-1 , -1], torch.float32, True),
    ])

    def forward(self, x, y):
        return torch.ops.aten.mse_loss(x, y, reduction=0)

@register_test_case(module_factory=lambda: MseLossNoReductionModule())
def MseLossNoReductionModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4), tu.rand(2, 4))


class MseLossMeanReductionModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1 , -1], torch.float32, True),
        ([-1 , -1], torch.float32, True),
    ])

    def forward(self, x, y):
        return torch.ops.aten.mse_loss(x, y, reduction=1)

@register_test_case(module_factory=lambda: MseLossMeanReductionModule())
def MseLossMeanReductionModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4), tu.rand(2, 4))


class MseLossSumReductionWithDifferentElemTypeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1 , -1], torch.float32, True),
        ([-1 , -1], torch.float64, True),
    ])

    def forward(self, x, y):
        return torch.ops.aten.mse_loss(x, y, reduction=2)

@register_test_case(module_factory=lambda: MseLossSumReductionWithDifferentElemTypeModule())
def MseLossSumReductionWithDifferentElemTypeModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4), tu.rand(2, 4).to(torch.float64))
