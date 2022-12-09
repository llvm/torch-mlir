# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================

class MeanModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([3, 4], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.mean(x)


@register_test_case(module_factory=lambda: MeanModule())
def MeanModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))

# ==============================================================================

class MeanDynamicSizesModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.mean(x)


@register_test_case(module_factory=lambda: MeanDynamicSizesModule())
def MeanDynamicSizesModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))

# ==============================================================================

class MeanDtypeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float64, True),
    ])
    def forward(self, x):
        return torch.ops.aten.mean(x, dtype=torch.float32)


@register_test_case(module_factory=lambda: MeanDtypeModule())
def MeanDtypeModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5).to(torch.float64))

# ==============================================================================

class MeanDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.mean(x, (0, 2))


@register_test_case(module_factory=lambda: MeanDimModule())
def MeanDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 7))

# ==============================================================================

class MeanDimDtypeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float64, True),
    ])
    def forward(self, x):
        return torch.ops.aten.mean(x, (0,), dtype=torch.float32)


@register_test_case(module_factory=lambda: MeanDimDtypeModule())
def MeanDimDtypeModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5).to(torch.float64))

# ==============================================================================

class MeanDimKeepdimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.mean(x, (1, 2), keepdim=True)


@register_test_case(module_factory=lambda: MeanDimKeepdimModule())
def MeanDimKeepdimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))

# ==============================================================================

class MeanDimAllReduceModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.mean(x, (0, 1, 2))


@register_test_case(module_factory=lambda: MeanDimAllReduceModule())
def MeanDimAllReduceModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))

# ==============================================================================

class MeanDimAllReduceKeepdimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.mean(x, (0, 1, 2), keepdim=True)


@register_test_case(module_factory=lambda: MeanDimAllReduceKeepdimModule())
def MeanDimAllReduceKeepdimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))

# ==============================================================================

class MeanDimNegativeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.mean(x, (-1, 1))


@register_test_case(module_factory=lambda: MeanDimNegativeModule())
def MeanDimNegativeModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))


# ==============================================================================

class MeanDimEmptyDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.mean(x, dim=[])


@register_test_case(module_factory=lambda: MeanDimEmptyDimModule())
def MeanDimEmptyDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))

# ==============================================================================

class MeanDimNoneDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.mean(x, dim=None)


@register_test_case(module_factory=lambda: MeanDimNoneDimModule())
def MeanDimNoneDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))

# ==============================================================================

class VarUnbiasedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.var(x, unbiased=True)

@register_test_case(module_factory=lambda: VarUnbiasedModule())
def VarUnbiasedModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4))

# ==============================================================================

class VarBiasedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.var(x, unbiased=False)

@register_test_case(module_factory=lambda: VarBiasedModule())
def VarBiasedModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4))

# ==============================================================================

class StdUnbiasedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.std(x, unbiased=True)

@register_test_case(module_factory=lambda: StdUnbiasedModule())
def StdUnbiasedModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4))

# ==============================================================================

class StdBiasedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.std(x, unbiased=False)

@register_test_case(module_factory=lambda: StdBiasedModule())
def StdBiasedModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4))


# ==============================================================================


class StdDimKeepDimFalseModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.std(x, dim=(1, 2), keepdim=False)


@register_test_case(module_factory=lambda: StdDimKeepDimFalseModule())
def StdDimKeepDimFalseModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))


# ==============================================================================


class StdDimKeepDimTrueModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.std(x, dim=(0, 1, 2), keepdim=True)


@register_test_case(module_factory=lambda: StdDimKeepDimFalseModule())
def StdDimKeepDimTrueModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))


# ==============================================================================


class StdDimBiasedModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.std(x, dim=(0, 2), unbiased=False)


@register_test_case(module_factory=lambda: StdDimBiasedModule())
def StdDimBiasedModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))


# ==============================================================================


class StdDimEmptyDimModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.std(x, dim=[], keepdim=False)


@register_test_case(module_factory=lambda: StdDimEmptyDimModule())
def StdDimEmptyDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))


# ==============================================================================


class StdDimNoneDimModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.std(x, dim=None, keepdim=False)


@register_test_case(module_factory=lambda: StdDimNoneDimModule())
def StdDimNoneDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))


# ==============================================================================


class VarDimModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.var(x, dim=(0, 2), keepdim=True)


@register_test_case(module_factory=lambda: VarDimModule())
def VarDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 7))


# ==============================================================================


class VarDimUnbiasedModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.var(x, dim=(0, 2), unbiased=True, keepdim=True)


@register_test_case(module_factory=lambda: VarDimUnbiasedModule())
def VarDimUnbiasedModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 7))


# ==============================================================================


class VarDimBiasedModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float64, True),
    ])
    def forward(self, x):
        return torch.ops.aten.var(x, dim=(0,1), unbiased=False, keepdim=True)


@register_test_case(module_factory=lambda: VarDimBiasedModule())
def VarDimBiasedModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5).to(torch.float64))


# ==============================================================================


class VarDimSingleDimModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float64, True),
    ])
    def forward(self, x):
        return torch.ops.aten.var(x, dim=(0,), keepdim=True)


@register_test_case(module_factory=lambda: VarDimSingleDimModule())
def VarDimSingleDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5).to(torch.float64))


# ==============================================================================


class VarDimMultiDimModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float64, True),
    ])
    def forward(self, x):
        return torch.ops.aten.var(x, dim=[0, 2], keepdim=False)


@register_test_case(module_factory=lambda: VarDimMultiDimModule())
def VarDimMultiDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5).to(torch.float64))


# ==============================================================================


class VarDimAllDimReduceModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.var(x, dim=(0, 1, 2), keepdim=True)


@register_test_case(module_factory=lambda: VarDimAllDimReduceModule())
def VarDimAllDimReduceModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))


# ==============================================================================


class VarDimNegativeModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.var(x, dim=(-1, 1), keepdim=True)


@register_test_case(module_factory=lambda: VarDimNegativeModule())
def VarDimNegativeModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))


# ==============================================================================


class VarDimEmptyDimModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.var(x, dim=[], keepdim=False)


@register_test_case(module_factory=lambda: VarDimEmptyDimModule())
def VarDimEmptyDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))


# ==============================================================================


class VarDimNoneDimModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.var(x, dim=None, keepdim=False)


@register_test_case(module_factory=lambda: VarDimNoneDimModule())
def VarDimNoneDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))


# ==============================================================================


class VarCorrectionModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.var(x, dim=None, correction=2)


@register_test_case(module_factory=lambda: VarCorrectionModule())
def VarCorrectionModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 7))


# ==============================================================================


class VarCorrectionSingleDimReduceModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.var(x, dim=[1], correction=1)


@register_test_case(module_factory=lambda: VarCorrectionSingleDimReduceModule())
def VarCorrectionSingleDimReduceModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 7))


# ==============================================================================


class VarCorrectionAllDimReduceModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.var(x,
                                  dim=[0, 1, 2],
                                  correction=10,
                                  keepdim=False)


@register_test_case(module_factory=lambda: VarCorrectionAllDimReduceModule())
def VarCorrectionAllDimReduceModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 7))


# ==============================================================================


class VarCorrectionKeepDimModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.var(x, dim=[0, 1], correction=None, keepdim=True)


@register_test_case(module_factory=lambda: VarCorrectionKeepDimModule())
def VarCorrectionKeepDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 7))


# ==============================================================================


class VarCorrectionNoneModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.var(x, dim=None, correction=None)


@register_test_case(module_factory=lambda: VarCorrectionNoneModule())
def VarCorrectionNoneModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 7))


# ==============================================================================


class VarCorrectionEmptyDimModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.var(x, dim=[], correction=2)


@register_test_case(module_factory=lambda: VarCorrectionEmptyDimModule())
def VarCorrectionEmptyDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 7))


# ==============================================================================


class VarCorrectionLargeInputModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.var(x, dim=[2, 3], correction=2)


@register_test_case(module_factory=lambda: VarCorrectionLargeInputModule())
def VarCorrectionLargeInputModule_basic(module, tu: TestUtils):
    module.forward(100 + tu.rand(3, 4, 1024, 8192))


# ==============================================================================


class VarMeanCorrectionModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.var_mean(x, dim=[1, 2], correction=2, keepdim=True)


@register_test_case(module_factory=lambda: VarMeanCorrectionModule())
def VarMeanCorrectionModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 7))


# ==============================================================================


class VarMeanCorrectionNoneModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.var_mean(x, dim=None, correction=None, keepdim=False)


@register_test_case(module_factory=lambda: VarMeanCorrectionNoneModule())
def VarMeanCorrectionNoneModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 7))


# ==============================================================================


class VarMeanUnbiasedModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.var_mean(x)


@register_test_case(module_factory=lambda: VarMeanUnbiasedModule())
def VarMeanUnbiasedModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 7))


# ==============================================================================


class VarMeanBiasedModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.var_mean(x, unbiased=False)


@register_test_case(module_factory=lambda: VarMeanBiasedModule())
def VarMeanBiasedModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 7))
