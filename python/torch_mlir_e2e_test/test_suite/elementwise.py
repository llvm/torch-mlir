# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.torchscript.framework import TestUtils
from torch_mlir_e2e_test.torchscript.registry import register_test_case
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export

# TODO: Support scalar !torch.int/!torch.float variants. Add support to
# ReduceOpVariants to implement them in terms of the tensor-only variants +
# torch.prim.NumToTensor.

# TODO: This is pretty verbose. Can we have a helper to reduce
# the boilerplate?

# ==============================================================================


class ElementwiseUnaryModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.tanh(a)


@register_test_case(module_factory=lambda: ElementwiseUnaryModule())
def ElementwiseUnaryModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


# ==============================================================================


class ElementwiseUnaryIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
    ])
    def forward(self, a):
        return torch.tanh(a)


@register_test_case(module_factory=lambda: ElementwiseUnaryIntModule())
def ElementwiseUnaryIntModule_basic(module, tu: TestUtils):
    module.forward(torch.randint(1, 10, (3, 4), dtype=torch.int32))


# ==============================================================================


class ElementwiseBinaryModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1], torch.float32, True),
    ])
    def forward(self, a, b):
        return a * b


@register_test_case(module_factory=lambda: ElementwiseBinaryModule())
def ElementwiseBinaryModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4), tu.rand(4))


# ==============================================================================


class ElementwiseBinaryStaticShapeModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([5, 4, 3, 3, 1], torch.float32, True),
        ([4, 3, 1, 2], torch.float32, True),
    ])
    def forward(self, a, b):
        return a * b


@register_test_case(
    module_factory=lambda: ElementwiseBinaryStaticShapeModule())
def ElementwiseBinaryStaticShapeModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 4, 3, 3, 1), tu.rand(4, 3, 1, 2))


# ==============================================================================


class ElementwiseTernaryModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
        ([-1], torch.float32, True),
    ])
    def forward(self, a, b, c):
        return torch.lerp(a, b, c)


@register_test_case(module_factory=lambda: ElementwiseTernaryModule())
def ElementwiseTernaryModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5), tu.rand(4, 5), tu.rand(5))


# ==============================================================================


class ElementwiseWhereSelfModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
        ([-1], torch.float32, True),
    ])
    def forward(self, a, b, c):
        return torch.where(a > 0.5, b, c)


@register_test_case(module_factory=lambda: ElementwiseWhereSelfModule())
def ElementwiseWhereSelfModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5), tu.rand(4, 5), tu.rand(5))


# ==============================================================================


class ElementwiseWhereScalarModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.where(a > 0.5, 4.0, 8.0)


@register_test_case(module_factory=lambda: ElementwiseWhereScalarModule())
def ElementwiseWhereScalarModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))


# ==============================================================================


class ElementwiseWhereScalarOtherModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float64, True),
        ([-1, -1], torch.float64, True),
    ])
    def forward(self, a, b):
        return torch.where(a > 0.5, b, 8.0)


@register_test_case(module_factory=lambda: ElementwiseWhereScalarOtherModule())
def ElementwiseWhereScalarOtherModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5).double(), tu.rand(4, 5).double())


# ==============================================================================


class ElementwiseWhereScalarSelfModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float64, True),
        ([-1, -1], torch.float64, True),
    ])
    def forward(self, a, b):
        return torch.where(a > 0.5, 4.0, b)


@register_test_case(module_factory=lambda: ElementwiseWhereScalarSelfModule())
def ElementwiseWhereScalarSelfModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5).double(), tu.rand(4, 5).double())


# ==============================================================================


# Addition is an interesting special case of a binary op, because under the hood
# it carries a third scalar "alpha" parameter, which needs special handling.
class ElementwiseAddModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.float32, True),
        ([], torch.float32, True),
    ])
    def forward(self, a, b):
        return a + b


@register_test_case(module_factory=lambda: ElementwiseAddModule())
def ElementwiseAddModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4), tu.rand())


# ==============================================================================


class ElementwiseUnsqueezeBroadcastModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.float32, True),
        ([], torch.float32, True),
    ])
    def forward(self, a, b):
        return a * b.unsqueeze(0)


@register_test_case(
    module_factory=lambda: ElementwiseUnsqueezeBroadcastModule())
def ElementwiseUnsqueezeBroadcastModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4), tu.rand())


# ==============================================================================


class ElementwiseUnsqueezeNegDimsModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        # As mentioned in `unsqueeze` docstring,
        # valid dim values are [-input.dim()-1, input.dim()+1).
        # This tests the lower bound
        return torch.unsqueeze(a, -3)


@register_test_case(module_factory=lambda: ElementwiseUnsqueezeNegDimsModule())
def ElementwiseUnsqueezeNegDimsModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 3))


# ==============================================================================


class ElementwiseFlattenBroadcastModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.float32, True),
        ([], torch.float32, True),
    ])
    def forward(self, a, b):
        return a * b.flatten(-1, -1)


@register_test_case(module_factory=lambda: ElementwiseFlattenBroadcastModule())
def ElementwiseFlattenBroadcastModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6), tu.rand())


# ==============================================================================


class ElementwiseReluModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.relu(x)


@register_test_case(module_factory=lambda: ElementwiseReluModule())
def ElementwiseReluModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 2) - 0.5)


# ==============================================================================


class ElementwiseLeakyReluModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.leaky_relu(x, negative_slope=0.1)


@register_test_case(module_factory=lambda: ElementwiseLeakyReluModule())
def ElementwiseLeakyReluModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 2) - 0.5)


# ==============================================================================


class ElementwiseGeluModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.gelu = torch.nn.GELU()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return self.gelu(x)


@register_test_case(module_factory=lambda: ElementwiseGeluModule())
def ElementwiseGeluModule_basic(module, tu: TestUtils):
    module.forward(2 * tu.rand(5, 3) - 0.5)


# ==============================================================================


class ElementwiseSigmoidModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.sigmoid(x)


@register_test_case(module_factory=lambda: ElementwiseSigmoidModule())
def ElementwiseSigmoidModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5))


# ==============================================================================


class ElementwiseSigmoidIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
    ])
    def forward(self, x):
        return torch.sigmoid(x)


@register_test_case(module_factory=lambda: ElementwiseSigmoidIntModule())
def ElementwiseSigmoidIntModule_basic(module, tu: TestUtils):
    module.forward(torch.randint(1, 10, (3, 5), dtype=torch.int32))


# ==============================================================================


class ElementwiseMinimumModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x, y):
        return torch.ops.aten.minimum(x, y)


@register_test_case(module_factory=lambda: ElementwiseMinimumModule())
def ElementwiseMinimumModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5), tu.rand(3, 5))


# ==============================================================================


class ElementwiseMinimumIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, x, y):
        return torch.ops.aten.minimum(x, y)


@register_test_case(module_factory=lambda: ElementwiseMinimumIntModule())
def ElementwiseMinimumIntModule_basic(module, tu: TestUtils):
    module.forward(torch.randint(10, (3, 5)), torch.randint(10, (3, 5)))


# ==============================================================================


class ElementwiseMaximumModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x, y):
        return torch.ops.aten.maximum(x, y)


@register_test_case(module_factory=lambda: ElementwiseMaximumModule())
def ElementwiseMaximumModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5), tu.rand(3, 5))


# ==============================================================================


class ElementwiseMaximumIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, x, y):
        return torch.ops.aten.maximum(x, y)


@register_test_case(module_factory=lambda: ElementwiseMaximumIntModule())
def ElementwiseMaximumIntModule_basic(module, tu: TestUtils):
    module.forward(torch.randint(10, (3, 5)), torch.randint(10, (3, 5)))


# ==============================================================================


class ElementwiseClampModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        float_min = torch.clamp(x, min=-2.0)
        int_min = torch.clamp(x, min=-3)
        float_max = torch.clamp(x, max=2.0)
        int_max = torch.clamp(x, max=3)
        both = torch.clamp(x, min=-5, max=5)
        return float_min, int_min, float_max, int_max, both


@register_test_case(module_factory=lambda: ElementwiseClampModule())
def ElementwiseClampModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5, low=-10, high=10))


# ==============================================================================


class ElementwiseClampMinModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        float_min = torch.ops.aten.clamp_min(x, min=-2.0)
        int_min = torch.ops.aten.clamp_min(x, min=2)
        min = torch.ops.aten.clamp_min(x, min=11.0)
        return float_min, int_min, min


@register_test_case(module_factory=lambda: ElementwiseClampMinModule())
def ElementwiseClampMinModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5, low=-10, high=10))


# ==============================================================================


class ElementwiseClampMaxModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        float_max = torch.ops.aten.clamp_max(x, max=2.0)
        int_max = torch.ops.aten.clamp_max(x, max=3)
        max = torch.ops.aten.clamp_max(x, max=-11.0)
        return float_max, int_max, max


@register_test_case(module_factory=lambda: ElementwiseClampMaxModule())
def ElementwiseClampMaxModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5, low=-10, high=10))


# ==============================================================================


class RsubFloatModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.rsub(x, 3.0, alpha=1.0)


@register_test_case(module_factory=lambda: RsubFloatModule())
def RsubFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


# ==============================================================================


class RsubFloatModule_noalpha(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.rsub(x, 2.0)


@register_test_case(module_factory=lambda: RsubFloatModule_noalpha())
def RsubFloatModule_noalpha_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


# ==============================================================================


class RsubIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, x):
        return torch.rsub(x, 2, alpha=3)


@register_test_case(module_factory=lambda: RsubIntModule())
def RsubIntModule_basic(module, tu: TestUtils):
    module.forward(torch.randint(100, (3, 4)))


# ==============================================================================


class RsubIntModule_noalpha(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, x):
        return torch.rsub(x, 2.)


@register_test_case(module_factory=lambda: RsubIntModule_noalpha())
def RsubIntModule_noalpha_basic(module, tu: TestUtils):
    module.forward(torch.randint(100, (3, 4)))


# ==============================================================================


class ElementwiseMulScalarIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, x):
        return torch.mul(x, 4)


@register_test_case(module_factory=lambda: ElementwiseMulScalarIntModule())
def ElementwiseMulScalarModule_int(module, tu: TestUtils):
    module.forward(torch.randint(10, (3, 4)))


# ==============================================================================


class ElementwiseMulScalarFloatModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.mul(x, 100.0)


@register_test_case(module_factory=lambda: ElementwiseMulScalarFloatModule())
def ElementwiseMulScalarModule_float(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


# ==============================================================================


class ElementwiseMulScalarModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
    ])
    def forward(self, x):
        return torch.mul(x, 8.0)


@register_test_case(module_factory=lambda: ElementwiseMulScalarModule())
def ElementwiseMulScalarModule_basic(module, tu: TestUtils):
    module.forward(torch.randint(10, (3, 4), dtype=torch.int32))


# ==============================================================================


class ElementwiseMulTensorFloatModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.float32, True),
        ([-1], torch.float64, True),
    ])
    def forward(self, a, b):
        return torch.mul(a, b)


@register_test_case(module_factory=lambda: ElementwiseMulTensorFloatModule())
def ElementwiseMulTensorFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4), tu.rand(4).type(torch.float64))


# ==============================================================================


class ElementwiseMulTensorIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.int32, True),
        ([-1], torch.int64, True),
    ])
    def forward(self, a, b):
        return torch.mul(a, b)


@register_test_case(module_factory=lambda: ElementwiseMulTensorIntModule())
def ElementwiseMulTensorIntModule_basic(module, tu: TestUtils):
    module.forward(
        torch.randint(10, [4]).type(torch.int32), torch.randint(10, [4]))


# ==============================================================================


class ElementwiseLogModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.log(a)


@register_test_case(module_factory=lambda: ElementwiseLogModule())
def ElementwiseLogModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


# ==============================================================================


class ElementwiseLogIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
    ])
    def forward(self, a):
        return torch.log(a)


@register_test_case(module_factory=lambda: ElementwiseLogIntModule())
def ElementwiseLogIntModule_basic(module, tu: TestUtils):
    module.forward(torch.randint(1, 10, (3, 4), dtype=torch.int32))


# ==============================================================================


class ElementwiseErfModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.ops.aten.erf(a)


@register_test_case(module_factory=lambda: ElementwiseErfModule())
def ElementwiseErfModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


# ==============================================================================


class ElementwiseErfIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
    ])
    def forward(self, a):
        return torch.ops.aten.erf(a)


@register_test_case(module_factory=lambda: ElementwiseErfIntModule())
def ElementwiseErfIntModule_basic(module, tu: TestUtils):
    module.forward(torch.randint(1, 10, (3, 4), dtype=torch.int32))


# ==============================================================================


class ElementwiseSqrtModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.sqrt(a)


@register_test_case(module_factory=lambda: ElementwiseSqrtModule())
def ElementwiseSqrtModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


# ==============================================================================


class ElementwiseSqrtIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
    ])
    def forward(self, a):
        return torch.sqrt(a)


@register_test_case(module_factory=lambda: ElementwiseSqrtIntModule())
def ElementwiseSqrtIntModule_basic(module, tu: TestUtils):
    module.forward(torch.randint(1, 10, (3, 4), dtype=torch.int32))


# ==============================================================================


class ElementwiseFloorModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.floor(a)


@register_test_case(module_factory=lambda: ElementwiseFloorModule())
def ElementwiseFloorModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


# ==============================================================================


class ElementwiseCeilModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.ceil(a)


@register_test_case(module_factory=lambda: ElementwiseCeilModule())
def ElementwiseCeilModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


# ==============================================================================


class ElementwisePowModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.pow(a, 2.0)


@register_test_case(module_factory=lambda: ElementwisePowModule())
def ElementwisePowModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


# ==============================================================================


class ElementwiseToDtypeF32ToI64Module(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1, -1], torch.float32, True)])
    def forward(self, x):
        return x.to(torch.int64)


@register_test_case(module_factory=lambda: ElementwiseToDtypeF32ToI64Module())
def ElementwiseToDtypeF32ToI64Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5))


# ==============================================================================


class ElementwiseToDtypeIdentityModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1, -1], torch.float32, True)])
    def forward(self, x):
        return x.to(torch.float32, False, False)


@register_test_case(module_factory=lambda: ElementwiseToDtypeIdentityModule())
def ElementwiseToDtypeIdentityModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5))


# ==============================================================================


class ElementwiseLog2Module(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.log2(a)


@register_test_case(module_factory=lambda: ElementwiseLog2Module())
def ElementwiseLog2Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


# ==============================================================================


class ElementwiseLog2IntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
    ])
    def forward(self, a):
        return torch.log2(a)


@register_test_case(module_factory=lambda: ElementwiseLog2IntModule())
def ElementwiseLog2IntModule_basic(module, tu: TestUtils):
    module.forward(torch.randint(1, 10, (3, 4), dtype=torch.int32))


# ==============================================================================


class ElementwiseRsqrtModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.rsqrt(a)


@register_test_case(module_factory=lambda: ElementwiseRsqrtModule())
def ElementwiseRsqrtModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


# ==============================================================================


class ElementwiseRsqrtIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
    ])
    def forward(self, a):
        return torch.rsqrt(a)


@register_test_case(module_factory=lambda: ElementwiseRsqrtIntModule())
def ElementwiseRsqrtIntModule_basic(module, tu: TestUtils):
    module.forward(torch.randint(1, 10, (3, 4), dtype=torch.int32))


# ==============================================================================


class ElementwiseAbsModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.abs(a)


@register_test_case(module_factory=lambda: ElementwiseAbsModule())
def ElementwiseAbsModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5, low=-1.0, high=1.0))


# ==============================================================================


class ElementwiseReciprocalModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.reciprocal(a)


@register_test_case(module_factory=lambda: ElementwiseReciprocalModule())
def ElementwiseReciprocalModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4))


# ==============================================================================


class ElementwiseReciprocalIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.int32, True),
    ])
    def forward(self, a):
        return torch.reciprocal(a)


@register_test_case(module_factory=lambda: ElementwiseReciprocalIntModule())
def ElementwiseReciprocalIntModule_basic(module, tu: TestUtils):
    module.forward(torch.randint(1, 10, (4,), dtype=torch.int32))


# ==============================================================================


class ElementwiseDivScalarModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.div(x, 10.0)


@register_test_case(module_factory=lambda: ElementwiseDivScalarModule())
def ElementwiseDivScalarModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


# ==============================================================================


class ElementwiseDivTensorFloatModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.float32, True),
        ([-1], torch.float64, True),
    ])
    def forward(self, a, b):
        return torch.div(a, b)


@register_test_case(module_factory=lambda: ElementwiseDivTensorFloatModule())
def ElementwiseDivTensorFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4), tu.rand(4).type(torch.float64))


# ==============================================================================


class ElementwiseDivRoundingModeTruncModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.float32, True),
        ([-1], torch.float64, True),
    ])
    def forward(self, a, b):
        return torch.div(a, b, rounding_mode="trunc")


@register_test_case(
    module_factory=lambda: ElementwiseDivRoundingModeTruncModule())
def ElementwiseDivRoundingModeTruncModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4), tu.rand(4).type(torch.float64))


class ElementwiseDivRoundingModeFloorModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float64, True),
    ])
    def forward(self, a, b):
        return torch.div(a, b, rounding_mode="floor")


@register_test_case(
    module_factory=lambda: ElementwiseDivRoundingModeFloorModule())
def ElementwiseDivRoundingModeFloorModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4), tu.rand(3, 4).type(torch.float64))


# ==============================================================================


class ElementwiseAndIntegerModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, x, y):
        return torch.bitwise_and(x, y)


@register_test_case(module_factory=lambda: ElementwiseAndIntegerModule())
def ElementwiseAndIntegerModule_basic(module, tu: TestUtils):
    module.forward(
        torch.randint(-10, 10, (3, 4)).to(torch.int32),
        torch.randint(-10, 10, (3, 4)))


# ==============================================================================


class ElementwiseSubScalarIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
    ])
    def forward(self, x):
        return torch.sub(x, 2.1, alpha=2)


@register_test_case(module_factory=lambda: ElementwiseSubScalarIntModule())
def ElementwiseSubScalarIntModule_basic(module, tu: TestUtils):
    module.forward(torch.randint(10, (3, 4), dtype=torch.int32))


# ==============================================================================


class ElementwiseSubScalarFloatModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.sub(x, 2.1)


@register_test_case(module_factory=lambda: ElementwiseSubScalarFloatModule())
def ElementwiseSubScalarFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


# ==============================================================================


class ElementwiseAddScalarInt64Module(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, x):
        return torch.add(x, 3.0)


@register_test_case(module_factory=lambda: ElementwiseAddScalarInt64Module())
def ElementwiseAddScalarInt64Module_basic(module, tu: TestUtils):
    module.forward(torch.randint(10, (3, 4)))


# ==============================================================================


class ElementwiseAddScalarIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
    ])
    def forward(self, x):
        return torch.add(x, 3.0)


@register_test_case(module_factory=lambda: ElementwiseAddScalarIntModule())
def ElementwiseAddScalarIntModule_basic(module, tu: TestUtils):
    module.forward(torch.randint(10, (2, 3), dtype=torch.int32))


# ==============================================================================


class ElementwiseAddScalarFloatModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.add(x, 3.0, alpha=2)


@register_test_case(module_factory=lambda: ElementwiseAddScalarFloatModule())
def ElementwiseAddScalarFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


# ==============================================================================


class ElementwiseCloneModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.clone(x)


@register_test_case(module_factory=lambda: ElementwiseCloneModule())
def ElementwiseCloneModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4))


# ==============================================================================


class ElementwiseCloneContiguousModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.clone(x, memory_format=torch.contiguous_format)


@register_test_case(module_factory=lambda: ElementwiseCloneContiguousModule())
def ElementwiseCloneContiguousModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4))


# ==============================================================================


class ElementwiseExpModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.exp(a)


@register_test_case(module_factory=lambda: ElementwiseExpModule())
def ElementwiseExpModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


# ==============================================================================


class ElementwiseExpIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
    ])
    def forward(self, a):
        return torch.exp(a)


@register_test_case(module_factory=lambda: ElementwiseExpIntModule())
def ElementwiseExpIntModule_basic(module, tu: TestUtils):
    module.forward(torch.randint(1, 10, (3, 4), dtype=torch.int32))


# ==============================================================================


class ElementwiseSinModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.sin(a)


@register_test_case(module_factory=lambda: ElementwiseSinModule())
def ElementwiseSinModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


# ==============================================================================


class ElementwiseSinIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
    ])
    def forward(self, a):
        return torch.sin(a)


@register_test_case(module_factory=lambda: ElementwiseSinIntModule())
def ElementwiseSinIntModule_basic(module, tu: TestUtils):
    module.forward(torch.randint(1, 10, (3, 4), dtype=torch.int32))


# ==============================================================================


class ElementwiseCosModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.cos(a)


@register_test_case(module_factory=lambda: ElementwiseCosModule())
def ElementwiseCosModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


# ==============================================================================


class ElementwiseCosIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
    ])
    def forward(self, a):
        return torch.cos(a)


@register_test_case(module_factory=lambda: ElementwiseCosIntModule())
def ElementwiseCosIntModule_basic(module, tu: TestUtils):
    module.forward(torch.randint(1, 10, (3, 4), dtype=torch.int32))


# ==============================================================================


class ElementwiseNegModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.neg(a)


@register_test_case(module_factory=lambda: ElementwiseNegModule())
def ElementwiseNegModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))

# ==============================================================================

class ElementwiseAtenLogicalOrOpModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.bool, True),
        ([-1], torch.bool, True),
    ])
    def forward(self, x, y):
        return torch.ops.aten.logical_or(x, y)

@register_test_case(module_factory=lambda: ElementwiseAtenLogicalOrOpModule())
def ElementwiseAtenLogicalOrOpModule_basic(module, tu: TestUtils):
    module.forward(torch.tensor([False, True]), torch.tensor([False, False]))

class ElementwiseAtenLogicalOrOpDiffArgs1Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.float64, True),
        ([-1], torch.int64, True),
    ])
    def forward(self, x, y):
        return torch.ops.aten.logical_or(x, y)

@register_test_case(module_factory=lambda: ElementwiseAtenLogicalOrOpDiffArgs1Module())
def ElementwiseAtenLogicalOrOpDiffArgs1Module_basic(module, tu: TestUtils):
    module.forward(torch.tensor([0.2, 0.1]), torch.tensor([0, 1]))

# ==============================================================================

class ElementwiseAtenLogicalOrOpDiffArgs2Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.bool, True),
        ([-1], torch.int64, True),
    ])
    def forward(self, x, y):
        return torch.ops.aten.logical_or(x, y)

@register_test_case(module_factory=lambda: ElementwiseAtenLogicalOrOpDiffArgs2Module())
def ElementwiseAtenLogicalOrOpDiffArgs2Module_basic(module, tu: TestUtils):
    module.forward(torch.tensor([True, False]), torch.tensor([0, 1]))

# ==============================================================================

class ElementwiseAtenLogicalOrOpDiffArgs3Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.int64, True),
        ([-1], torch.bool, True),
    ])
    def forward(self, x, y):
        return torch.ops.aten.logical_or(x, y)

@register_test_case(module_factory=lambda: ElementwiseAtenLogicalOrOpDiffArgs3Module())
def ElementwiseAtenLogicalOrOpDiffArgs3Module_basic(module, tu: TestUtils):
    module.forward(torch.tensor([1, 2]), torch.tensor([False, True]))

# ==============================================================================

class ElementwiseAtenLogicalOrOpRandomModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.int64, True),
        ([-1, -1, -1, -1], torch.int64, True),
    ])
    def forward(self, x, y):
        return torch.ops.aten.logical_or(x, y)

@register_test_case(module_factory=lambda: ElementwiseAtenLogicalOrOpRandomModule())
def ElementwiseAtenLogicalOrOpRandomModule_basic(module, tu: TestUtils):
    module.forward(torch.randint(3, 10, (2, 3, 4, 5)), torch.randint(10, 100, (2, 3, 4, 5)))

# ==============================================================================

class ElementwiseAtenLogicalOrOpRandomFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, x, y):
        return torch.ops.aten.logical_or(x, y)

@register_test_case(module_factory=lambda: ElementwiseAtenLogicalOrOpRandomFloatModule())
def ElementwiseAtenLogicalOrOpRandomFloatModule_basic(module, tu: TestUtils):
    module.forward(torch.rand(2, 3, 3, 5), torch.rand(2, 3, 3, 5))

# ==============================================================================

class ElementwiseAtenLogicalOrOpNegativeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.int64, True),
        ([-1, -1, -1, -1], torch.int64, True),
    ])
    def forward(self, x, y):
        return torch.ops.aten.logical_or(x, y)

@register_test_case(module_factory=lambda: ElementwiseAtenLogicalOrOpNegativeModule())
def ElementwiseAtenLogicalOrOpNegativeModule_basic(module, tu: TestUtils):
    module.forward(torch.neg(torch.randint(3, 10, (2, 3, 4, 5))), torch.neg(torch.randint(10, 100, (2, 3, 4, 5))))

# ==============================================================================

class ElementwiseAtenLogicalOrOpBrodcastModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    @export
    @annotate_args([
        None,
        ([-1],     torch.int64, True),
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, x, y):
        return torch.ops.aten.logical_or(x, y)

@register_test_case(module_factory=lambda: ElementwiseAtenLogicalOrOpBrodcastModule())
def ElementwiseAtenLogicalOrOpBrodcastModule_basic(module, tu: TestUtils):
    module.forward(torch.randint(3, (3,)), torch.randint(3, (4, 3)))


# ==============================================================================


class ElementwiseAtenFloorDivideModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x, y):
        return torch.ops.aten.floor_divide(x, y)


@register_test_case(module_factory=lambda: ElementwiseAtenFloorDivideModule())
def ElementwiseAtenFloorDivideModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 3), tu.rand(4, 3))


class ElementwiseAtenFloorDivideBroadcastModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.float32, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x, y):
        return torch.ops.aten.floor_divide(x, y)


@register_test_case(
    module_factory=lambda: ElementwiseAtenFloorDivideBroadcastModule())
def ElementwiseAtenFloorDivideBroadcastModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3), tu.rand(4, 3))


# ==============================================================================


class AtenTriuModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.triu(x)


@register_test_case(module_factory=lambda: AtenTriuModule())
def AtenTriuModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 8, 3, 4, 3))


# ==============================================================================


class AtenTriuWithPosDiagonalModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.triu(x, diagonal=2)


@register_test_case(module_factory=lambda: AtenTriuWithPosDiagonalModule())
def AtenTriuWithPosDiagonalModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(9, 4, 3))


# ==============================================================================


class AtenTriuWithNegDiagonalModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.triu(x, diagonal=-4)


@register_test_case(module_factory=lambda: AtenTriuWithNegDiagonalModule())
def AtenTriuWithNegDiagonalModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 1, 5, 9))
