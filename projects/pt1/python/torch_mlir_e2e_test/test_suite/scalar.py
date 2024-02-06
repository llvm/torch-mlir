# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================


class AddIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([], torch.int64, True),
        ([], torch.int64, True),
    ])
    def forward(self, lhs, rhs):
        return int(lhs) + int(rhs)


@register_test_case(module_factory=lambda: AddIntModule())
def AddIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(low=-100, high=100), tu.randint(low=-100, high=100))


# ==============================================================================


class SubIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([], torch.int64, True),
        ([], torch.int64, True),
    ])
    def forward(self, lhs, rhs):
        return int(lhs) - int(rhs)


@register_test_case(module_factory=lambda: SubIntModule())
def SubIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(low=-100, high=100), tu.randint(low=-100, high=100))


# ==============================================================================


class SubFloatModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([], torch.float64, True),
        ([], torch.float64, True),
    ])
    def forward(self, lhs, rhs):
        return float(lhs) - float(rhs)


@register_test_case(module_factory=lambda: SubFloatModule())
def SubFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand().double(), tu.rand().double())


# ==============================================================================

class MulFloatModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([], torch.float64, True),
        ([], torch.float64, True),
    ])
    def forward(self, lhs, rhs):
        return float(lhs) * float(rhs)


@register_test_case(module_factory=lambda: MulFloatModule())
def MulFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand().double(), tu.rand().double())


# ==============================================================================


class MulIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([], torch.int64, True),
        ([], torch.int64, True),
    ])
    def forward(self, lhs, rhs):
        return int(lhs) * int(rhs)


@register_test_case(module_factory=lambda: MulIntModule())
def MulIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(low=-100, high=100), tu.randint(low=-100, high=100))


# ==============================================================================


class DivIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([], torch.int64, True),
        ([], torch.int64, True),
    ])
    def forward(self, lhs, rhs):
        # Cast the result to float to make e2e test baseline result to be a float.
        # Without the cast, baseline result is a Tensor which is unexpected.
        return float(torch.ops.aten.div(int(lhs), int(rhs)))


@register_test_case(module_factory=lambda: DivIntModule())
def DivIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(low=-10, high=10), tu.randint(low=3, high=10))


# ==============================================================================


class DivFloatModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([], torch.float64, True),
        ([], torch.float64, True),
    ])
    def forward(self, lhs, rhs):
        return float(lhs) / float(rhs)


@register_test_case(module_factory=lambda: DivFloatModule())
def DivFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand().double(), tu.rand().double())


# ==============================================================================


class CeilFloatModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([], torch.float64, True),
        ([], torch.float64, True),
    ])
    def forward(self, lhs, rhs):
        sub = float(lhs) - float(rhs)
        # Cast the result to int to make e2e test baseline result to be an int.
        # Without the cast, baseline result is a Tensor which is unexpected see
        # https://github.com/llvm/torch-mlir/issues/842
        # TODO: Investigate the root cause of baseline returning a Tensor
        # without the int cast and remove the cast.
        return int(torch.ops.aten.ceil(float(sub)))


@register_test_case(module_factory=lambda: CeilFloatModule())
def CeilFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand().double(), tu.rand().double())


# ==============================================================================


class SqrtIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([], torch.int64, True),
    ])
    def forward(self, a):
        return float(torch.ops.aten.sqrt(int(a)))


@register_test_case(module_factory=lambda: SqrtIntModule())
def SqrtIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(high=10))


class SqrtIntConstantModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        return float(torch.ops.aten.sqrt(5))


@register_test_case(module_factory=lambda: SqrtIntConstantModule())
def SqrtIntConstantModule_basic(module, tu: TestUtils):
    module.forward()


# ==============================================================================


class BoolFloatFalseModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([], torch.float64, True),
    ])
    def forward(self, a):
        sub = float(a) - float(a)
        return bool(torch.ops.aten.Bool(float(sub)))


@register_test_case(module_factory=lambda: BoolFloatFalseModule())
def BoolFloatFalseModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(low=0.5).double())


class BoolFloatTrueModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([], torch.float64, True),
    ])
    def forward(self, a):
        return bool(torch.ops.aten.Bool(float(a)))


@register_test_case(module_factory=lambda: BoolFloatTrueModule())
def BoolFloatTrueModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(low=0.5).double())


class BoolFloatConstantModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        return bool(torch.ops.aten.Bool(5.0))


@register_test_case(module_factory=lambda: BoolFloatConstantModule())
def BoolFloatConstantModule_basic(module, tu: TestUtils):
    module.forward()


# ==============================================================================


class BoolIntFalseModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([], torch.int64, True),
    ])
    def forward(self, a):
        sub = int(a) - int(a)
        return bool(torch.ops.aten.Bool(int(sub)))


@register_test_case(module_factory=lambda: BoolIntFalseModule())
def BoolIntFalseModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(low=1, high=100))


class BoolIntTrueModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([], torch.int64, True),
    ])
    def forward(self, a):
        return bool(torch.ops.aten.Bool(int(a)))


@register_test_case(module_factory=lambda: BoolIntTrueModule())
def BoolIntTrueModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(low=1, high=100))


class BoolIntConstantModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        return bool(torch.ops.aten.Bool(5))


@register_test_case(module_factory=lambda: BoolIntConstantModule())
def BoolIntConstantModule_basic(module, tu: TestUtils):
    module.forward()

# ==============================================================================

class AtenIntBoolOpModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([], torch.bool, True),
    ])
    def forward(self, x):
        return int(torch.ops.aten.Int(x))


@register_test_case(module_factory=lambda: AtenIntBoolOpModule())
def AtenIntBoolOpModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(low=0, high=2).bool())


class AtenIntBoolOpConstTrueModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        return int(torch.ops.aten.Int(True))


@register_test_case(module_factory=lambda: AtenIntBoolOpConstTrueModule())
def AtenIntBoolOpConstTrueModule_basic(module, tu: TestUtils):
    module.forward()


class AtenIntBoolOpConstFalseModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        return int(torch.ops.aten.Int(False))


@register_test_case(module_factory=lambda: AtenIntBoolOpConstFalseModule())
def AtenIntBoolOpConstFalseModule_basic(module, tu: TestUtils):
    module.forward()

# ==============================================================================

class AtenIntTensorByteDtypeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([], torch.uint8, True),
    ])

    def forward(self, val):
        return int(val)

@register_test_case(module_factory=lambda: AtenIntTensorByteDtypeModule())
def AtenIntTensorByteDtypeModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(low=-100, high=100).to(dtype=torch.uint8))


# ==============================================================================

class AtenIntTensorCharDtypeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([], torch.int8, True),
    ])

    def forward(self, val):
        return int(val)

@register_test_case(module_factory=lambda: AtenIntTensorCharDtypeModule())
def AtenIntTensorCharDtypeModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(low=-100, high=100).to(dtype=torch.int8))

# ==============================================================================

class AtenItemIntOpModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([], torch.int8, True),
    ])

    def forward(self, val):
        return int(val)

@register_test_case(module_factory=lambda: AtenItemIntOpModule())
def AtenItemIntOpModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(low=-100, high=100).to(dtype=torch.int8))

# ==============================================================================

class AtenItemFpOpModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([], torch.float, True),
    ])

    def forward(self, val):
        return float(val)

@register_test_case(module_factory=lambda: AtenItemFpOpModule())
def AtenItemFpOpModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1))
