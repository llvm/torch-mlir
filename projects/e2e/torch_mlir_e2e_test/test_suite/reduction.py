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
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float64, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.bool, True),
        ]
    )
    def forward(self, a):
        return torch.sum(a)


@register_test_case(module_factory=lambda: ReduceSumElementTypeBoolModule())
def ReduceSumElementTypeBoolModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, 5, high=2).to(torch.bool))


# ==============================================================================


class PrimsSumFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.prims.sum(a, (0, 1))


@register_test_case(module_factory=lambda: PrimsSumFloatModule())
def PrimsSumFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))


# ==============================================================================


class ReduceProdFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.prod(a)


@register_test_case(module_factory=lambda: ReduceProdFloatModule())
def ReduceProdFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))


# ==============================================================================


class ReduceProdDtypeFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float64, True),
        ]
    )
    def forward(self, a):
        return torch.prod(a, dtype=torch.float32)


@register_test_case(module_factory=lambda: ReduceProdDtypeFloatModule())
def ReduceProdDtypeFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5).to(torch.float64))


# ==============================================================================


class ReduceProdElementTypeBoolModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.bool, True),
        ]
    )
    def forward(self, a):
        return torch.prod(a)


@register_test_case(module_factory=lambda: ReduceProdElementTypeBoolModule())
def ReduceProdElementTypeBoolModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, 5, high=2).to(torch.bool))


# ==============================================================================


class ReduceAllFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.all(a)


@register_test_case(module_factory=lambda: ReduceAllFloatModule())
def ReduceAllFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))


class ReduceAllDimFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.all(a, dim=0)


@register_test_case(module_factory=lambda: ReduceAllDimFloatModule())
def ReduceAllDimFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))


# ==============================================================================


class ReduceAllIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.int32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.all(a)


@register_test_case(module_factory=lambda: ReduceAllIntModule())
def ReduceAllIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, 5, high=2).to(torch.int32))


# ==============================================================================


class ReduceAllBoolModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.bool, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.all(a)


@register_test_case(module_factory=lambda: ReduceAllBoolModule())
def ReduceAllBoolModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, high=2).to(torch.bool))


# ==============================================================================


class ReduceAnyFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.any(a)


@register_test_case(module_factory=lambda: ReduceAnyFloatModule())
def ReduceAnyFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))


class ReduceAnyDimFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.any(a, dim=0)


@register_test_case(module_factory=lambda: ReduceAnyDimFloatModule())
def ReduceAnyDimFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))


class ReduceAnyDimsFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.any(a, dim=[0, 1])


@register_test_case(module_factory=lambda: ReduceAnyDimsFloatModule())
def ReduceAnyDimsFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))


# ==============================================================================


class ReduceAnyIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.int32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.any(a)


@register_test_case(module_factory=lambda: ReduceAnyIntModule())
def ReduceAnyIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, 5, high=2).to(torch.int32))


# ==============================================================================


class ReduceAnyBoolModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.bool, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.any(a)


@register_test_case(module_factory=lambda: ReduceAnyBoolModule())
def ReduceAnyBoolModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, high=2).to(torch.bool))


# ==============================================================================


class ReduceSumDimIntListFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float64, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([1, 12, 7, 7], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.sum(a, dim=(-1), keepdim=True)


@register_test_case(
    module_factory=lambda: ReduceSumDimIntListKeepDimNegativeDimStaticModule()
)
def ReduceSumDimIntListKeepDimNegativeDimStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 12, 7, 7))


# ==============================================================================


class ReduceSumDimIntListEmptyDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1], torch.bool, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.int64, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.int64, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.int32, True),
        ]
    )
    def forward(self, a):
        return torch.sum(a, dtype=torch.int64)


@register_test_case(module_factory=lambda: ReduceSumDtypeIntModule())
def ReduceSumDtypeIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, 5, high=100).to(torch.int32))


# ==============================================================================


class ReduceProdUnsignedIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.int64, True),
        ]
    )
    def forward(self, a):
        return torch.prod(a)


@register_test_case(module_factory=lambda: ReduceProdUnsignedIntModule())
def ReduceProdUnsignedIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, 5, low=0, high=100))


# ==============================================================================


class ReduceProdSignedIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.int64, True),
        ]
    )
    def forward(self, a):
        return torch.prod(a)


@register_test_case(module_factory=lambda: ReduceProdSignedIntModule())
def ReduceProdSignedIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, 5, low=-100, high=100))


# ==============================================================================


class ReduceProdDtypeIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.int32, True),
        ]
    )
    def forward(self, a):
        return torch.prod(a, dtype=torch.int64)


@register_test_case(module_factory=lambda: ReduceProdDtypeIntModule())
def ReduceProdDtypeIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, 5, high=100).to(torch.int32))


# ==============================================================================


class ReduceSumDimIntListIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.int64, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.int32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.int64, True),
        ]
    )
    def forward(self, a):
        return torch.sum(a, (1, 2), keepdim=True)


@register_test_case(module_factory=lambda: ReduceSumDimIntListKeepDimIntModule())
def ReduceSumDimIntListKeepDimIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, 5, high=100))


# ==============================================================================


class ReduceProdDimIntFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.prod(a, 1, dtype=torch.float32)


@register_test_case(module_factory=lambda: ReduceProdDimIntFloatModule())
def ReduceProdDimIntFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5).to(torch.float32))


# ==============================================================================


class ReduceAllDimEmpty(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.all(a, dim=0, keepdim=False)


@register_test_case(module_factory=lambda: ReduceAllDimEmpty())
def ReduceAllDimEmpty_basic(module, tu: TestUtils):
    module.forward(torch.tensor([]))


# ==============================================================================


class ReduceAllDimFloat(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.all(a, dim=1, keepdim=True)


@register_test_case(module_factory=lambda: ReduceAllDimFloat())
def ReduceAllDimFloat_basic(module, tu: TestUtils):
    module.forward(torch.tensor([[5.0, 1e-6, -5.0], [0, 5.0, 0]]))


# ==============================================================================


class ReduceAllDimInt(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.int32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.all(a, dim=1, keepdim=True)


@register_test_case(module_factory=lambda: ReduceAllDimInt())
def ReduceAllDimInt_basic(module, tu: TestUtils):
    module.forward(torch.tensor([[5, -5, 0], [5, 1e10, 5]]).to(torch.int32))


# ==============================================================================


class ReduceAllDimBool(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.bool, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.all(a, dim=1, keepdim=False)


@register_test_case(module_factory=lambda: ReduceAllDimBool())
def ReduceAllDimBool_basic(module, tu: TestUtils):
    module.forward(torch.tensor([[True, False, True], [True, True, True]]))


# ==============================================================================


class ReduceMaxAlongDim(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float64, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.max(a, 1)[0]


@register_test_case(module_factory=lambda: ReduceMaxAlongDim())
def ReduceMaxAlongDim_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5).to(torch.float64))


# ==============================================================================


class ReduceMinAlongDim(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float64, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.min(a, 1)[0]


@register_test_case(module_factory=lambda: ReduceMinAlongDim())
def ReduceMinAlongDim_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5).to(torch.float64))


class ReduceMinAlongDimSignedInt(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.int64, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.min(a, 1)


@register_test_case(module_factory=lambda: ReduceMinAlongDimSignedInt())
def ReduceMinAlongDimSignedInt_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, 5, low=-100, high=100))


# ==============================================================================


class ReduceMinAlongDimUnsignedInt(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.uint8, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.min(a, 1)


@register_test_case(module_factory=lambda: ReduceMinAlongDimUnsignedInt())
def ReduceMinAlongDimUnsignedInt_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, 5, low=-100, high=100).to(torch.uint8))


# ==============================================================================


class ReduceMinAlongDimNegative(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float64, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.min(a, 1)[0]


@register_test_case(module_factory=lambda: ReduceMinAlongDimNegative())
def ReduceMinAlongDimNegative_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5, low=-10, high=10).to(torch.float64))


# ==============================================================================


class ReduceMinKeepDim(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float64, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.min(a, 1, keepdim=True)[1]


@register_test_case(module_factory=lambda: ReduceMinKeepDim())
def ReduceMinKeepDim_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5).to(torch.float64))


# ==============================================================================


class ReduceMinKeepDimReturnBoth(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.min(a, 1, keepdim=True)


@register_test_case(module_factory=lambda: ReduceMinKeepDimReturnBoth())
def ReduceMinKeepDimReturnBoth_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5, low=-10, high=-5))


# ==============================================================================


class ReduceMaxAlongDimSignedInt(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.int64, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.max(a, 1)


@register_test_case(module_factory=lambda: ReduceMaxAlongDimSignedInt())
def ReduceMaxAlongDimSignedInt_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, 5, low=-100, high=100))


# ==============================================================================


class ReduceMaxAlongDimUnsignedInt(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.uint8, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.max(a, 1)


@register_test_case(module_factory=lambda: ReduceMaxAlongDimUnsignedInt())
def ReduceMaxAlongDimUnsignedInt_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, 5, low=-100, high=100).to(torch.uint8))


# ==============================================================================


class ReduceMaxAlongDimNegative(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float64, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float64, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.int64, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.int64, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.amax(a, (0, 2))


@register_test_case(module_factory=lambda: ReduceAmaxMultiDim())
def ReduceAmaxMultiDim_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5, high=100))


# ==============================================================================


class ReduceAmaxEmptyDim(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.amax(a, dim=())


@register_test_case(module_factory=lambda: ReduceAmaxEmptyDim())
def ReduceAmaxEmptyDim_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5, high=100))


# ==============================================================================


class ReduceAmaxOutOfOrderDim(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.amax(a, (0, 2), keepdim=True)


@register_test_case(module_factory=lambda: ReduceAmaxKeepDim())
def ReduceAmaxKeepDim_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5, high=100))


# ==============================================================================


class ReduceAminSingleDim(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.amin(a, 1)


@register_test_case(module_factory=lambda: ReduceAminSingleDim())
def ReduceAminSingleDim_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5, high=100))


# ==============================================================================


class ReduceAminmaxSingleDim(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.aminmax(a, dim=1)


@register_test_case(module_factory=lambda: ReduceAminmaxSingleDim())
def ReduceAminmaxSingleDim_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5, high=100))


# ==============================================================================


class ReduceAminmaxAllDims(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.aminmax(a)


@register_test_case(module_factory=lambda: ReduceAminmaxAllDims())
def ReduceAminmaxAllDims_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5, high=100))


# ==============================================================================


class ReduceMinFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.min(a)


@register_test_case(module_factory=lambda: ReduceMinFloatModule())
def ReduceMinFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))


# ==============================================================================


class ReduceMinSignedIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.int64, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.min(a)


@register_test_case(module_factory=lambda: ReduceMinSignedIntModule())
def ReduceMinSignedIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, 5, low=-100, high=100))


# ==============================================================================


class ReduceMinUnsignedIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.int64, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.min(a)


@register_test_case(module_factory=lambda: ReduceMinUnsignedIntModule())
def ReduceMinUnsignedIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, 5, high=100))


# ==============================================================================


class ArgminModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.argmin(a)


@register_test_case(module_factory=lambda: ArgminModule())
def ArgminModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


# ==============================================================================


class ArgminIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.int64, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.argmin(a)


@register_test_case(module_factory=lambda: ArgminIntModule())
def ArgminIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, low=-100, high=100))


@register_test_case(module_factory=lambda: ArgminIntModule())
def ArgminIntModule_multiple_mins(module, tu: TestUtils):
    # To cover the special case that the minimal value occurs more than once.
    # The pytorch convention is here to consider the first occurence as the argmin.
    module.forward(torch.full((3, 4), tu.randint(1).item(), dtype=torch.int64))


# ==============================================================================


class ArgminWithDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.argmin(a, dim=1)


@register_test_case(module_factory=lambda: ArgminWithDimModule())
def ArgminModule_with_dim(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))


# ==============================================================================


class ArgminKeepDimsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.argmin(a, 0, True)


@register_test_case(module_factory=lambda: ArgminKeepDimsModule())
def ArgminModule_keepDim(module, tu: TestUtils):
    module.forward(tu.rand(4, 6))


# ==============================================================================


class ArgmaxModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.argmax(a)


@register_test_case(module_factory=lambda: ArgmaxModule())
def ArgmaxModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


# ==============================================================================


class ArgmaxKeepdimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.argmax(a, keepdim=True)


@register_test_case(module_factory=lambda: ArgmaxKeepdimModule())
def ArgmaxKeepdimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


# ==============================================================================


class ArgmaxIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.int64, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.argmax(a)


@register_test_case(module_factory=lambda: ArgmaxIntModule())
def ArgmaxIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, low=-100, high=100))


@register_test_case(module_factory=lambda: ArgmaxIntModule())
def ArgmaxIntModule_multiple_maxs(module, tu: TestUtils):
    # To cover the special case that the maximal value occurs more than once.
    # The pytorch convention is here to consider the first occurence as the argmax.
    module.forward(torch.full((3, 4), tu.randint(1).item(), dtype=torch.int64))


# ==============================================================================


class ArgmaxWithDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.argmax(a, dim=1)


@register_test_case(module_factory=lambda: ArgmaxWithDimModule())
def ArgmaxModule_with_dim(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))


# ==============================================================================


class ArgmaxKeepDimsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.argmax(a, 0, True)


@register_test_case(module_factory=lambda: ArgmaxKeepDimsModule())
def ArgmaxModule_keepDim(module, tu: TestUtils):
    module.forward(tu.rand(4, 6))


# ==============================================================================


class ReduceL1NormModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.linalg.vector_norm(a, dim=0, ord=1)


@register_test_case(module_factory=lambda: ReduceL1NormModule())
def ReduceL1NormModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))


# ==============================================================================


class ReduceL1NormWithDTypeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.linalg.vector_norm(a, dim=0, ord=1, dtype=torch.float64)


@register_test_case(module_factory=lambda: ReduceL1NormWithDTypeModule())
def ReduceL1NormWithDTypeModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5).to(torch.float32))


# ==============================================================================


class ReduceL1NormComplexModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.cfloat, True),
        ]
    )
    def forward(self, a):
        return torch.linalg.vector_norm(a, dim=0, ord=1)


@register_test_case(module_factory=lambda: ReduceL1NormComplexModule())
def ReduceL1NormComplexModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5).to(torch.cfloat))


# ==============================================================================


class ReduceL2NormModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.linalg.vector_norm(a, dim=0)


@register_test_case(module_factory=lambda: ReduceL2NormModule())
def ReduceL2NormModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))


# ==============================================================================


class ReduceL2NormComplexModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.cdouble, True),
        ]
    )
    def forward(self, a):
        return torch.linalg.vector_norm(a, dim=0)


@register_test_case(module_factory=lambda: ReduceL2NormComplexModule())
def ReduceL2NormComplexModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5).to(torch.cdouble))


# ==============================================================================


class ReduceLN3NormModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.linalg.vector_norm(a, dim=0, ord=-3)


@register_test_case(module_factory=lambda: ReduceLN3NormModule())
def ReduceLN3NormModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))


# ==============================================================================


class ReduceL3NormAllDimsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.linalg.vector_norm(a, dim=None, ord=3)


@register_test_case(module_factory=lambda: ReduceL3NormAllDimsModule())
def ReduceL3NormAllDimsModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))


# ==============================================================================


class ReduceL3NormKeepDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.linalg.vector_norm(a, keepdim=True, ord=3)


@register_test_case(module_factory=lambda: ReduceL3NormKeepDimModule())
def ReduceL3NormKeepDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))


# ==============================================================================


class ReduceL3NormKeepDimComplexModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.complex128, True),
        ]
    )
    def forward(self, a):
        return torch.linalg.vector_norm(a, keepdim=True, ord=3)


@register_test_case(module_factory=lambda: ReduceL3NormKeepDimComplexModule())
def ReduceL3NormKeepDimComplexModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5).to(torch.complex128))


# ==============================================================================


class NormScalarModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.p = 3.0

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.norm(a, self.p)


@register_test_case(module_factory=lambda: NormScalarModule())
def NormScalarModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))


# ==============================================================================


class NormScalarComplexModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.p = 3.0

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.complex64, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.norm(a, self.p)


@register_test_case(module_factory=lambda: NormScalarComplexModule())
def NormScalarComplexModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5).to(torch.complex64))


# ==============================================================================


class NormScalarOptDimModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.p = 3.0

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.norm(a, self.p, dim=[0, 1], keepdim=False)


@register_test_case(module_factory=lambda: NormScalarOptDimModule())
def NormScalarOptDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))


# ==============================================================================


class NormScalarOptDimKeepDimModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.p = 3.0

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.norm(a, self.p, dim=[0, 1], keepdim=True)


@register_test_case(module_factory=lambda: NormScalarOptDimKeepDimModule())
def NormScalarOptDimKeepDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))


# ==============================================================================


class NormScalarOptDimKeepDimComplexModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.p = 3.0

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.cfloat, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.norm(a, self.p, dim=[0, 1], keepdim=True)


@register_test_case(module_factory=lambda: NormScalarOptDimKeepDimComplexModule())
def NormScalarOptDimKeepDimComplexModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5).to(torch.cfloat))


# ==============================================================================


class ReduceFrobeniusNormModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.frobenius_norm(a, dim=[0, 1], keepdim=False)


@register_test_case(module_factory=lambda: ReduceFrobeniusNormModule())
def ReduceFrobeniusNormModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))


# ==============================================================================


class ReduceFrobeniusNormKeepDimModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.frobenius_norm(a, dim=[0, 1], keepdim=True)


@register_test_case(module_factory=lambda: ReduceFrobeniusNormKeepDimModule())
def ReduceFrobeniusNormKeepDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))


# ==============================================================================


class ReduceFrobeniusNormComplexModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.cdouble, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.frobenius_norm(a, dim=[0, 1], keepdim=False)


@register_test_case(module_factory=lambda: ReduceFrobeniusNormComplexModule())
def ReduceFrobeniusNormComplexModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5).to(torch.cdouble))


# ==============================================================================


class LinalgVectorNormModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.linalg_vector_norm(a, ord=3.0, dim=[0, 1], keepdim=False)


@register_test_case(module_factory=lambda: LinalgVectorNormModule())
def LinalgVectorNormModule_basic(module, tu: TestUtils):
    module.forward(torch.rand(3, 4, 5))


# ==============================================================================


class LinalgVectorNormKeepDimModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.linalg_vector_norm(a, ord=3.0, dim=[0, 1], keepdim=True)


@register_test_case(module_factory=lambda: LinalgVectorNormKeepDimModule())
def LinalgVectorNormKeepDimModule_basic(module, tu: TestUtils):
    module.forward(torch.rand(3, 4, 5))


# ==============================================================================


class LinalgVectorNormComplexModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.complex128, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.linalg_vector_norm(a, ord=3.0, dim=[0, 1], keepdim=False)


@register_test_case(module_factory=lambda: LinalgVectorNormComplexModule())
def LinalgVectorNormComplexModule_basic(module, tu: TestUtils):
    module.forward(torch.rand(3, 4, 5).to(torch.complex128))


# ==============================================================================


class LinalgNormModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.linalg_norm(a, ord=None, dim=[0], keepdim=False)


@register_test_case(module_factory=lambda: LinalgNormModule())
def LinalgNormModule_basic(module, tu: TestUtils):
    module.forward(torch.rand(3, 4, 5))


# ==============================================================================


class LinalgNormKeepDimModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.linalg_norm(a, ord=None, dim=[0], keepdim=True)


@register_test_case(module_factory=lambda: LinalgNormKeepDimModule())
def LinalgNormKeepDimModule_basic(module, tu: TestUtils):
    module.forward(torch.rand(3, 4, 5))


# ==============================================================================


class LinalgNormKeepDimComplexModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.complex64, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.linalg_norm(a, ord=None, dim=[0], keepdim=True)


@register_test_case(module_factory=lambda: LinalgNormKeepDimComplexModule())
def LinalgNormKeepDimComplexModule_basic(module, tu: TestUtils):
    module.forward(torch.rand(3, 4, 5).to(torch.complex64))


# ==============================================================================


class MseLossNoReductionModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, x, y):
        return torch.ops.aten.mse_loss(x, y, reduction=0)


@register_test_case(module_factory=lambda: MseLossNoReductionModule())
def MseLossNoReductionModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4), tu.rand(2, 4))


# ==============================================================================


class MseLossMeanReductionModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, x, y):
        return torch.ops.aten.mse_loss(x, y, reduction=1)


@register_test_case(module_factory=lambda: MseLossMeanReductionModule())
def MseLossMeanReductionModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4), tu.rand(2, 4))


# ==============================================================================


class MseLossSumReductionWithDifferentElemTypeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
            ([-1, -1], torch.float64, True),
        ]
    )
    def forward(self, x, y):
        return torch.ops.aten.mse_loss(x, y, reduction=2)


@register_test_case(
    module_factory=lambda: MseLossSumReductionWithDifferentElemTypeModule()
)
def MseLossSumReductionWithDifferentElemTypeModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4), tu.rand(2, 4).to(torch.float64))


# ==============================================================================


class L1LossNoReductionModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 4], torch.float32, True),
            ([2, 4], torch.float32, True),
        ]
    )
    def forward(self, x, y):
        return torch.ops.aten.l1_loss(x, y, reduction=0)


@register_test_case(module_factory=lambda: L1LossNoReductionModule())
def L1LossNoReductionModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4), tu.rand(2, 4))


# ==============================================================================


class L1LossMeanReductionModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 4], torch.float32, True),
            ([2, 4], torch.float32, True),
        ]
    )
    def forward(self, x, y):
        return torch.ops.aten.l1_loss(x, y, reduction=1)


@register_test_case(module_factory=lambda: L1LossMeanReductionModule())
def L1LossMeanReductionModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4), tu.rand(2, 4))


# ==============================================================================


class L1LossSumReductionModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 4], torch.float32, True),
            ([2, 4], torch.float32, True),
        ]
    )
    def forward(self, x, y):
        return torch.ops.aten.l1_loss(x, y, reduction=2)


@register_test_case(module_factory=lambda: L1LossSumReductionModule())
def L1LossSumReductionModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4), tu.rand(2, 4))


# ==============================================================================


class CrossEntropyLossModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
            (
                [
                    -1,
                ],
                torch.int64,
                True,
            ),
        ]
    )
    def forward(self, input, target):
        return torch.ops.aten.cross_entropy_loss(input, target)


@register_test_case(module_factory=lambda: CrossEntropyLossModule())
def CrossEntropyLossModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(8, 2), tu.randint(8, high=2))


class CrossEntropyLossNoReductionModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
            (
                [
                    -1,
                ],
                torch.int64,
                True,
            ),
        ]
    )
    def forward(self, input, target):
        return torch.ops.aten.cross_entropy_loss(input, target, reduction=0)


@register_test_case(module_factory=lambda: CrossEntropyLossNoReductionModule())
def CrossEntropyLossNoReductionModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(8, 2), tu.randint(8, high=2))


class BinaryCrossEntropyWithLogitsStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([8, 2], torch.float32, True),
            ([8, 2], torch.float32, True),
        ]
    )
    def forward(self, input, target):
        return torch.ops.aten.binary_cross_entropy_with_logits(
            input, target, reduction=0
        )


@register_test_case(module_factory=lambda: BinaryCrossEntropyWithLogitsStaticModule())
def BinaryCrossEntropyWithLogitsStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(8, 2), tu.rand(8, 2))


# ==============================================================================


class TraceModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.trace(a)


@register_test_case(module_factory=lambda: TraceModule())
def TraceModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 3))


@register_test_case(module_factory=lambda: TraceModule())
def TraceModule_nonsquare(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


@register_test_case(module_factory=lambda: TraceModule())
def TraceModule_empty(module, tu: TestUtils):
    module.forward(torch.empty(0, 0))


# ==============================================================================


class TraceIntModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.int64, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.trace(a)


@register_test_case(module_factory=lambda: TraceIntModule())
def TraceSignedIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(2, 2, low=-10, high=10))


@register_test_case(module_factory=lambda: TraceIntModule())
def TraceUnsignedIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(2, 2, low=0, high=10))


@register_test_case(module_factory=lambda: TraceIntModule())
def TraceUnsignedIntModule_empty(module, tu: TestUtils):
    module.forward(tu.randint(0, 0, low=0, high=10))


# ==============================================================================


class CountNonzeroModuleI64(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 2

    @export
    @annotate_args([None, ([2, 3, 4], torch.int64, True)])
    def forward(self, x):
        return torch.ops.aten.count_nonzero(x, self.dim)


@register_test_case(module_factory=lambda: CountNonzeroModuleI64())
def CountNonzeroModuleI64_basic(module, tu: TestUtils):
    module.forward(tu.randint(2, 3, 4, low=-2, high=2))


class CountNonzeroModuleF32(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = -3

    @export
    @annotate_args([None, ([2, 3, 4], torch.float32, True)])
    def forward(self, x):
        return torch.ops.aten.count_nonzero(x, self.dim)


@register_test_case(module_factory=lambda: CountNonzeroModuleF32())
def CountNonzeroModuleF32_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4))


class CountNonzeroModuleBool(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([2, 3, 4], torch.bool, True)])
    def forward(self, x):
        return torch.ops.aten.count_nonzero(x)


@register_test_case(module_factory=lambda: CountNonzeroModuleBool())
def CountNonzeroModuleBool_Basic(module, tu: TestUtils):
    module.forward(tu.randint(2, 3, 4, low=0, high=2).to(torch.bool))


# ==============================================================================


class CountNonzeroDimIntListModuleF32(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = [-1, 1]

    @export
    @annotate_args([None, ([2, 3, 4], torch.float32, True)])
    def forward(self, x):
        return torch.ops.aten.count_nonzero(x, self.dim)


@register_test_case(module_factory=lambda: CountNonzeroDimIntListModuleF32())
def CountNonzeroDimIntListModuleF32_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4))


class CountNonzeroDimIntListModuleI64(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = [-2, -0, -1]

    @export
    @annotate_args([None, ([2, 3, 4], torch.int64, True)])
    def forward(self, x):
        return torch.ops.aten.count_nonzero(x, self.dim)


@register_test_case(module_factory=lambda: CountNonzeroDimIntListModuleI64())
def CountNonzeroDimIntListModuleI64_basic(module, tu: TestUtils):
    module.forward(tu.randint(2, 3, 4))


class CountNonzeroDimIntListModuleBool(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = [1]

    @export
    @annotate_args([None, ([2, 3, 4], torch.bool, True)])
    def forward(self, x):
        return torch.ops.aten.count_nonzero(x, self.dim)


@register_test_case(module_factory=lambda: CountNonzeroDimIntListModuleBool())
def CountNonzeroDimIntListModuleBool_Basic(module, tu: TestUtils):
    module.forward(tu.randint(2, 3, 4, low=0, high=2).to(torch.bool))
