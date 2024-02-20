# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

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
    module.forward(tu.randint(3, 4, low=1, high=10).to(torch.int32))


# ==============================================================================


class ElementwiseCoshModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.cosh(a)


@register_test_case(module_factory=lambda: ElementwiseCoshModule())
def ElementwiseCoshModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


# ==============================================================================


class ElementwiseCoshIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
    ])
    def forward(self, a):
        return torch.cosh(a)


@register_test_case(module_factory=lambda: ElementwiseCoshIntModule())
def ElementwiseCoshIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, low=1, high=10).to(torch.int32))


# ==============================================================================


class ElementwiseAcoshModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.acosh(a)


@register_test_case(module_factory=lambda: ElementwiseAcoshModule())
def ElementwiseAcoshModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


# ==============================================================================


class ElementwiseAcoshIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
    ])
    def forward(self, a):
        return torch.acosh(a)


@register_test_case(module_factory=lambda: ElementwiseAcoshIntModule())
def ElementwiseAcoshIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, low=1, high=10).to(torch.int32))


# ==============================================================================


class ElementwiseAsinModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.asin(a)


@register_test_case(module_factory=lambda: ElementwiseAsinModule())
def ElementwiseAsinModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


# ==============================================================================


class ElementwiseAsinIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
    ])
    def forward(self, a):
        return torch.asin(a)


@register_test_case(module_factory=lambda: ElementwiseAsinIntModule())
def ElementwiseAsinIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, low=1, high=10).to(torch.int32))


# ==============================================================================


class ElementwiseAsinhModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.asinh(a)


@register_test_case(module_factory=lambda: ElementwiseAsinhModule())
def ElementwiseAsinhModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


# ==============================================================================


class ElementwiseAsinhIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
    ])
    def forward(self, a):
        return torch.asinh(a)


@register_test_case(module_factory=lambda: ElementwiseAsinhIntModule())
def ElementwiseAsinhIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, low=1, high=10).to(torch.int32))


# ==============================================================================


class ElementwiseAtanhModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.atanh(a)


@register_test_case(module_factory=lambda: ElementwiseAtanhModule())
def ElementwiseAtanhModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


# ==============================================================================


class ElementwiseAtanhIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
    ])
    def forward(self, a):
        return torch.atanh(a)


@register_test_case(module_factory=lambda: ElementwiseAtanhIntModule())
def ElementwiseAtanhIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, low=1, high=10).to(torch.int32))


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


class ElementwiseAtenWhereSelfModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([1, 1, 5, 5], torch.bool, True),
        ([1, 12, 5, 5], torch.float32, True),
        ([], torch.float32, True),
    ])
    def forward(self, a, b, c):
        return torch.ops.aten.where(a, b, c)


@register_test_case(module_factory=lambda: ElementwiseAtenWhereSelfModule())
def ElementwiseAtenWhereSelfModule_basic(module, tu: TestUtils):
    module.forward(torch.zeros(1, 1, 5, 5, dtype=torch.bool), tu.rand(1, 12, 5, 5), tu.rand())


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


class ElementwiseWhereScalarOtherStaticModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([3, 4, 5], torch.float64, True),
        ([4, 5], torch.float64, True),
    ])
    def forward(self, a, b):
        return torch.where(a > 0.5, b, 8)


@register_test_case(module_factory=lambda: ElementwiseWhereScalarOtherStaticModule())
def ElementwiseWhereScalarOtherStaticModule_basic(module, tu: TestUtils):
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


class ElementwiseWhereScalarSelfStaticModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([3, 4, 5], torch.float64, True),
        ([4, 5], torch.float64, True),
    ])
    def forward(self, a, b):
        return torch.where(a > 0.5, 4.0, b)


@register_test_case(module_factory=lambda: ElementwiseWhereScalarSelfStaticModule())
def ElementwiseWhereScalarSelfStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5).double(), tu.rand(4, 5).double())


# ==============================================================================


class ElementwiseNanToNumModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([3, 4], torch.float32, True)
    ])
    def forward(self, a):
        return torch.ops.aten.nan_to_num(a, 0.0, 1.0, -1.0)

@register_test_case(module_factory=lambda: ElementwiseNanToNumModule())
def ElementwiseNanToNumModule_Basic(module, tu: TestUtils):
    module.forward(torch.tensor(
        [
            [float('nan'), 0.0, float('nan'), 0.0],
            [float('inf'), 0.0, float('inf'), 0.0],
            [float('-inf'), 0.0, float('-inf'), 0.0]
        ]
    ))


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
    module.forward(tu.rand(4, 2, low=-1))


# ==============================================================================


class ElementwiseRelu6Module(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.relu6(x)


@register_test_case(module_factory=lambda: ElementwiseRelu6Module())
def ElementwiseRelu6Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 2, low=-1))


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
    module.forward(tu.rand(4, 2, low=-1))


class ElementwiseLeakyReluStaticModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([4, 5, 6], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.leaky_relu(x, negative_slope=0.1)


@register_test_case(module_factory=lambda: ElementwiseLeakyReluStaticModule())
def ElementwiseLeakyReluStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 5, 6, low=-1))


# ==============================================================================


class ElementwiseLerpScalarIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a, b):
        return torch.ops.aten.lerp(a, b, weight=2)

@register_test_case(module_factory=lambda: ElementwiseLerpScalarIntModule())
def ElementwiseLerpScalarIntModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5,3), tu.rand(5,3))


class ElementwiseLerpScalarFloatModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a, b):
        return torch.ops.aten.lerp(a, b, weight=0.5)

@register_test_case(module_factory=lambda: ElementwiseLerpScalarFloatModule())
def ElementwiseLerpScalarFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5,3), tu.rand(5,3))


# ==============================================================================


class ElementwiseEluNonDefaultModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.elu(x, scale=1.5, alpha=2.0, input_scale=3.0)

@register_test_case(module_factory=lambda: ElementwiseEluNonDefaultModule())
def ElementwiseEluNonDefaultModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5,3, low=-1, high=1))


# ==============================================================================


class ElementwiseEluModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.elu(x)

@register_test_case(module_factory=lambda: ElementwiseEluModule())
def ElementwiseEluModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5,3, low=-1, high=1))


# ==============================================================================


class ElementwisePreluModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1, -1], torch.float32, True),
        ([-1], torch.float32, True),
    ])
    def forward(self, x, weight):
        return torch.ops.aten.prelu(x, weight)

@register_test_case(module_factory=lambda: ElementwisePreluModule())
def ElementwisePreluModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 4, 3, 2, 1, low=-1, high=1), tu.rand(1) )


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
    module.forward(tu.rand(5, 3, low=-0.5, high=0.5))


# ==============================================================================


class ElementwiseSeluModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.selu(x)

@register_test_case(module_factory=lambda: ElementwiseSeluModule())
def ElementwiseSeluModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 3, low=-1, high=1))


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
    module.forward(tu.randint(3, 5, low=1, high=10).to(torch.int32))


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
    module.forward(tu.randint(3, 5, high=10), tu.randint(3, 5, high=10))


# ==============================================================================


class ElementwiseMinOtherModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x, y):
        return x.min(y)


@register_test_case(module_factory=lambda: ElementwiseMinOtherModule())
def ElementwiseMinOtherModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5), tu.rand(3, 5))


# ==============================================================================


class ElementwiseMinOtherIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, x, y):
        return x.min(y)


@register_test_case(module_factory=lambda: ElementwiseMinOtherIntModule())
def ElementwiseMinOtherIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 5, high=10), tu.randint(3, 5, high=10))


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
    module.forward(tu.randint(3, 5, high=10), tu.randint(3, 5, high=10))


# ==============================================================================


class ElementwiseMaxOtherModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x, y):
        return x.max(y)


@register_test_case(module_factory=lambda: ElementwiseMaxOtherModule())
def ElementwiseMaxOtherModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5), tu.rand(3, 5))


# ==============================================================================


class ElementwiseMaxOtherIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, x, y):
        return x.max(y)


@register_test_case(module_factory=lambda: ElementwiseMaxOtherIntModule())
def ElementwiseMaxOtherIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 5, high=10), tu.randint(3, 5, high=10))


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


class ElementwiseClampTensorFloatModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([], torch.float32, True),
        ([], torch.float32, True),
    ])
    def forward(self, x, min, max):
        min_clamp = torch.clamp(x, min)
        max_clamp = torch.clamp(x, max=max)
        both_clamp = torch.clamp(x, min=min, max=max)
        return min_clamp, max_clamp, both_clamp


@register_test_case(module_factory=lambda: ElementwiseClampTensorFloatModule())
def ElementwiseClampTensorFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5, low=-10, high=10), torch.tensor([-5.0]), torch.tensor([5.0]))


# ==============================================================================


class ElementwiseClampTensorIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
        ([], torch.int64, True),
        ([], torch.int64, True),
    ])
    def forward(self, x, min, max):
        min_clamp = torch.clamp(x, min)
        max_clamp = torch.clamp(x, max=max)
        both_clamp = torch.clamp(x, min=min, max=max)
        return min_clamp, max_clamp, both_clamp


@register_test_case(module_factory=lambda: ElementwiseClampTensorIntModule())
def ElementwiseClampTensorIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 5, low=-10, high=10), torch.tensor([-5]), torch.tensor([5]))


# ==============================================================================


class ElementwiseClampTensorInt8Module(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int8, True)
    ])
    def forward(self, x):
        min = -5
        max = 5
        min_clamp = torch.clamp(x, min)
        max_clamp = torch.clamp(x, max=max)
        both_clamp = torch.clamp(x, min=min, max=max)
        return min_clamp, max_clamp, both_clamp


@register_test_case(module_factory=lambda: ElementwiseClampTensorInt8Module())
def ElementwiseClampTensorInt8Module_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 5, low=-10, high=10, dtype=torch.int8))


# ==============================================================================



class ElementwiseClampMinTensorFloatModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([], torch.float32, True),
    ])
    def forward(self, x, min):
        return torch.ops.aten.clamp_min(x, min=min)


@register_test_case(module_factory=lambda: ElementwiseClampMinTensorFloatModule())
def ElementwiseClampMinTensorFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5, low=-10, high=10), torch.tensor([-5.0]))


# ==============================================================================


class ElementwiseClampMinTensorIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
        ([], torch.int64, True),
    ])
    def forward(self, x, min):
        return torch.ops.aten.clamp_min(x, min=min)


@register_test_case(module_factory=lambda: ElementwiseClampMinTensorIntModule())
def ElementwiseClampMinTensorIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 5, low=-10, high=10), torch.tensor([-5]))


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
    module.forward(tu.randint(3, 4, high=100))


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
    module.forward(tu.randint(3, 4, high=100))

# ==============================================================================


class RsubInt0d_NumToTensor_Module(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        x = torch.ops.prim.NumToTensor(5)
        return torch.rsub(x, 2)


@register_test_case(module_factory=lambda: RsubInt0d_NumToTensor_Module())
def RsubInt0d_NumToTensor_Module_basic(module, tu: TestUtils):
    module.forward()

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
    module.forward(tu.randint(3, 4, high=10))


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
    module.forward(tu.randint(3, 4, high=10).to(torch.int32))


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
        tu.randint(4, high=10).type(torch.int32), tu.randint(4, high=10))


# ==============================================================================

class ElementwiseMulTensorComplexModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.complex64, True),
        ([-1], torch.complex64, True),
    ])
    def forward(self, a, b):
        return torch.mul(a, b)


@register_test_case(module_factory=lambda: ElementwiseMulTensorComplexModule())
def ElementwiseMulTensorComplexModule_basic(module, tu: TestUtils):
    module.forward(
        tu.randint(4, high=10).type(torch.complex64), tu.randint(4, high=10).type(torch.complex64))


# ==============================================================================

class ElementwiseMishModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.mish(x)


@register_test_case(module_factory=lambda: ElementwiseMishModule())
def ElementwiseMishModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 3, low=-3.0, high=3.0))


# ==============================================================================


class ElementwiseAtanTensorFloatModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.atan(a)


@register_test_case(module_factory=lambda: ElementwiseAtanTensorFloatModule())
def ElementwiseAtanTensorFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 4))


# ==============================================================================


class ElementwiseAtanTensorIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.int32, True),
    ])
    def forward(self, a):
        return torch.atan(a)


@register_test_case(module_factory=lambda: ElementwiseAtanTensorIntModule())
def ElementwiseAtanTensorIntModule_basic(module, tu: TestUtils):
    module.forward(
        tu.randint(4, low=1, high=10).type(torch.int32))


# ==============================================================================


class ElementwiseAtan2TensorFloatModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a, b):
        return torch.atan2(a, b)


@register_test_case(module_factory=lambda: ElementwiseAtan2TensorFloatModule())
def ElementwiseAtan2TensorFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 4), tu.rand(4, 4))


# ==============================================================================


class ElementwiseAtan2TensorIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.int32, True),
        ([-1], torch.int64, True),
    ])
    def forward(self, a, b):
        return torch.atan2(a, b)


@register_test_case(module_factory=lambda: ElementwiseAtan2TensorIntModule())
def ElementwiseAtan2TensorIntModule_basic(module, tu: TestUtils):
    module.forward(
        tu.randint(4, low=1, high=10).type(torch.int32), tu.randint(4, low=1, high=10))


# ==============================================================================


class ElementwiseAtan2FloatIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
        ([-1, -1], torch.float64, True),
    ])
    def forward(self, a, b):
        return torch.atan2(a, b)


@register_test_case(module_factory=lambda: ElementwiseAtan2FloatIntModule())
def ElementwiseAtan2FloatIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(4, 4, low=1, high=10).to(torch.int32),
                   tu.rand(4, 4).double())


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
    module.forward(tu.randint(3, 4, low=1, high=10).to(torch.int32))

# ==============================================================================


class ElementwiseLog1pModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.log1p(a)


@register_test_case(module_factory=lambda: ElementwiseLog1pModule())
def ElementwiseLog1pModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


# ==============================================================================


class ElementwiseLogitModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.logit(a, eps=1e-7)


@register_test_case(module_factory=lambda: ElementwiseLogitModule())
def ElementwiseLogitModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


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
    module.forward(tu.randint(3, 4, low=1, high=10).to(torch.int32))


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
    module.forward(tu.randint(3, 4, low=1, high=10).to(torch.int32))


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

class ElementwiseFloorIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
    ])
    def forward(self, a):
        return torch.floor(a)


@register_test_case(module_factory=lambda: ElementwiseFloorIntModule())
def ElementwiseFloorIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, low=-10, high=10).to(torch.int32))


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


class ElementwiseSignModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.ops.aten.sign(a)


@register_test_case(module_factory=lambda: ElementwiseSignModule())
def ElementwiseSignModule_basic(module, tu: TestUtils):
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


class ElementwisePowTensorModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a, b):
        return torch.pow(a, b)


@register_test_case(module_factory=lambda: ElementwisePowTensorModule())
def ElementwisePowTensorModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4), tu.rand(3, 4))


# ==============================================================================


class ElementwisePowTensorStaticModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([3, 4], torch.float32, True),
        ([1, 1], torch.float32, True),
    ])
    def forward(self, a, b):
        return torch.pow(a, b)


@register_test_case(module_factory=lambda: ElementwisePowTensorStaticModule())
def ElementwisePowTensorStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4), tu.rand(1, 1))


# ==============================================================================


class ElementwisePowTensorBroadcastModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, 1], torch.float32, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a, b):
        return torch.pow(a, b)


@register_test_case(module_factory=lambda: ElementwisePowTensorBroadcastModule())
def ElementwisePowTensorBroadcastModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 1), tu.rand(3, 4))


# ==============================================================================


class ElementwisePowTensorBroadcastStaticModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([3, 1], torch.float32, True),
        ([3, 4], torch.float32, True),
    ])
    def forward(self, a, b):
        return torch.pow(a, b)


@register_test_case(module_factory=lambda: ElementwisePowTensorBroadcastStaticModule())
def ElementwisePowTensorBroadcastStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 1), tu.rand(3, 4))


# ==============================================================================


class ElementwisePowScalarModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([3, 4], torch.float32, True),
    ])
    def forward(self, exp):
        return torch.pow(2.0, exp)


@register_test_case(module_factory=lambda: ElementwisePowScalarModule())
def ElementwisePowScalarModule_basic(module, tu: TestUtils):
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


class ElementwiseToDtypeI64ToI8Module(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1, -1], torch.int64, True)])
    def forward(self, x):
        return x.to(torch.int8)


@register_test_case(module_factory=lambda: ElementwiseToDtypeI64ToI8Module())
def ElementwiseToDtypeI64ToI8Module_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, low=-100, high=100))


# ==============================================================================


class ElementwiseToDtypeI64ToUI8Module(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1, -1], torch.int64, True)])
    def forward(self, x):
        return x.to(torch.uint8)


@register_test_case(module_factory=lambda: ElementwiseToDtypeI64ToUI8Module())
def ElementwiseToDtypeI64ToUI8Module_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, low=-100, high=100))


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
    module.forward(tu.randint(3, 4, low=1, high=10).to(torch.int32))


# ==============================================================================

class ElementwiseLog10Module(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.log10(a)


@register_test_case(module_factory=lambda: ElementwiseLog10Module())
def ElementwiseLog10Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


# ==============================================================================

class ElementwiseLog10IntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
    ])
    def forward(self, a):
        return torch.log10(a)


@register_test_case(module_factory=lambda: ElementwiseLog10IntModule())
def ElementwiseLog10IntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, low=1, high=10).to(torch.int32))


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
    module.forward(tu.randint(3, 4, low=1, high=10).to(torch.int32))


# ==============================================================================


class ElementwiseAbsFloatModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.abs(a)


@register_test_case(module_factory=lambda: ElementwiseAbsFloatModule())
def ElementwiseAbsFloatModule_basic(module, tu: TestUtils):
    module.forward(torch.tensor([[[-1.0, 0.0, 1.0]]]))


# ==============================================================================


class ElementwiseAbsIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.int64, True),
    ])
    def forward(self, a):
        return torch.abs(a)


@register_test_case(module_factory=lambda: ElementwiseAbsIntModule())
def ElementwiseAbsIntModule_basic(module, tu: TestUtils):
    module.forward(torch.tensor([[[-1, 0, 1]]]))


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
    module.forward(tu.randint(4, low=1, high=10).to(torch.int32))


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


class ElementwiseAtenDivIntScalarModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, x):
        return torch.ops.aten.div(x, 128)


@register_test_case(module_factory=lambda: ElementwiseAtenDivIntScalarModule())
def ElementwiseAtenDivIntScalarModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4))

# ==============================================================================


class ElementwiseRemainderScalarModule_Int_Float(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.int32, True),
    ])
    def forward(self, x):
        return torch.remainder(x, 2.0)


@register_test_case(module_factory=lambda: ElementwiseRemainderScalarModule_Int_Float())
def ElementwiseRemainderScalarModule_Int_Float_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, high=10).to(torch.int32))


# ==============================================================================


class ElementwiseRemainderScalarModule_Float(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.remainder(x, 2.0)


@register_test_case(module_factory=lambda: ElementwiseRemainderScalarModule_Float())
def ElementwiseRemainderScalarModule_Float_basic(module, tu: TestUtils):
    module.forward(tu.rand(10, 3))


# ==============================================================================

class ElementwiseRemainderScalarModule_Int(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
    ])
    def forward(self, x):
        return torch.remainder(x, 2)


@register_test_case(module_factory=lambda: ElementwiseRemainderScalarModule_Int())
def ElementwiseRemainderScalarModule_Int_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 2, high=10).to(torch.int32))

# ==============================================================================

class ElementwiseRemainderScalarModule_Bool(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.bool, True),
    ])
    def forward(self, x):
        return torch.remainder(x, 2)


@register_test_case(module_factory=lambda: ElementwiseRemainderScalarModule_Bool())
def ElementwiseRemainderScalarModule_Bool_basic(module, tu: TestUtils):
    module.forward(torch.tensor([True, False, True, True, True]))


# ==============================================================================


class ElementwiseRemainderTensorModule_Int_Float(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a, b):
        return torch.remainder(a, b)


@register_test_case(module_factory=lambda: ElementwiseRemainderTensorModule_Int_Float())
def ElementwiseRemainderTensorModule_Int_Float_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, high=10).to(torch.int32), tu.rand(3, 4, high=10))


# ==============================================================================


class ElementwiseRemainderTensorModule_Float(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a, b):
        return torch.remainder(a, b)


@register_test_case(module_factory=lambda: ElementwiseRemainderTensorModule_Float())
def ElementwiseRemainderTensorModule_Float_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, high=10), tu.rand(3, 4, high=10))


# ==============================================================================

class ElementwiseRemainderTensorModule_Int(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
        ([-1, -1], torch.int32, True),
    ])
    def forward(self, a, b):
        return torch.remainder(a, b)


@register_test_case(module_factory=lambda: ElementwiseRemainderTensorModule_Int())
def ElementwiseRemainderTensorModule_Int_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, high=10, dtype=torch.int32), tu.randint(3, 4, high=10, dtype=torch.int32))

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


class ElementwiseBitwiseAndModule(torch.nn.Module):

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


@register_test_case(module_factory=lambda: ElementwiseBitwiseAndModule())
def ElementwiseBitwiseAndModule_basic(module, tu: TestUtils):
    module.forward(
        tu.randint(3, 4, low=-10, high=10).to(torch.int32),
        tu.randint(3, 4, low=-10, high=10))


# ==============================================================================


class ElementwiseBitwiseAndStaticShapeModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([3, 4], torch.int32, True),
        ([4], torch.int64, True),
    ])
    def forward(self, x, y):
        return torch.bitwise_and(x, y)


@register_test_case(module_factory=lambda: ElementwiseBitwiseAndStaticShapeModule())
def ElementwiseBitwiseAndStaticShapeModule_basic(module, tu: TestUtils):
    module.forward(
        tu.randint(3, 4, low=-10, high=10).to(torch.int32),
        tu.randint(4, low=-10, high=10))


# ==============================================================================


class ElementwiseBitwiseOrModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, x, y):
        return torch.bitwise_or(x, y)


@register_test_case(module_factory=lambda: ElementwiseBitwiseOrModule())
def ElementwiseBitwiseOrModule_basic(module, tu: TestUtils):
    module.forward(
        tu.randint(3, 4, low=-10, high=10).to(torch.int32),
        tu.randint(3, 4, low=-10, high=10))


# ==============================================================================


class ElementwiseBitwiseOrStaticShapeModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([3, 4], torch.int32, True),
        ([4], torch.int64, True),
    ])
    def forward(self, x, y):
        return torch.bitwise_or(x, y)


@register_test_case(module_factory=lambda: ElementwiseBitwiseOrStaticShapeModule())
def ElementwiseBitwiseOrStaticShapeModule_basic(module, tu: TestUtils):
    module.forward(
        tu.randint(3, 4, low=-10, high=10).to(torch.int32),
        tu.randint(4, low=-10, high=10))


# ==============================================================================


class ElementwiseOrTensorModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, x, y):
        return torch.ops.aten.__or__(x, y)


@register_test_case(module_factory=lambda: ElementwiseOrTensorModule())
def ElementwiseOrTensorModule_basic(module, tu: TestUtils):
    module.forward(
        tu.randint(3, 4, low=-10, high=10).to(torch.int32),
        tu.randint(3, 4, low=-10, high=10))


# ==============================================================================


class ElementwiseOrTensorStaticShapeModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([3, 4], torch.int32, True),
        ([4], torch.int64, True),
    ])
    def forward(self, x, y):
        return torch.ops.aten.__or__(x, y)


@register_test_case(module_factory=lambda: ElementwiseOrTensorStaticShapeModule())
def ElementwiseOrTensorStaticShapeModule_basic(module, tu: TestUtils):
    module.forward(
        tu.randint(3, 4, low=-10, high=10).to(torch.int32),
        tu.randint(4, low=-10, high=10))


# ==============================================================================


class ElementwiseBitwiseXorModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, x, y):
        return torch.bitwise_xor(x, y)


@register_test_case(module_factory=lambda: ElementwiseBitwiseXorModule())
def ElementwiseBitwiseXorModule_basic(module, tu: TestUtils):
    module.forward(
        tu.randint(3, 4, low=-10, high=10).to(torch.int32),
        tu.randint(3, 4, low=-10, high=10))


# ==============================================================================


class ElementwiseBitwiseXorStaticShapeModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([3, 4], torch.int32, True),
        ([4], torch.int64, True),
    ])
    def forward(self, x, y):
        return torch.bitwise_xor(x, y)


@register_test_case(module_factory=lambda: ElementwiseBitwiseXorStaticShapeModule())
def ElementwiseBitwiseXorStaticShapeModule_basic(module, tu: TestUtils):
    module.forward(
        tu.randint(3, 4, low=-10, high=10).to(torch.int32),
        tu.randint(4, low=-10, high=10))


# ==============================================================================


class ElementwiseBitwiseNotInt64Module(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, x):
        return torch.bitwise_not(x)


@register_test_case(module_factory=lambda: ElementwiseBitwiseNotInt64Module())
def ElementwiseBitwiseNotInt64Module_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, low=-10, high=10))


# ==============================================================================


class ElementwiseBitwiseNotInt32Module(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
    ])
    def forward(self, x):
        return torch.bitwise_not(x)


@register_test_case(module_factory=lambda: ElementwiseBitwiseNotInt32Module())
def ElementwiseBitwiseNotInt32Module_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, low=-10, high=10).to(torch.int32))


# ==============================================================================


class ElementwiseSubTensorInt8Module(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int8, True),
        ([-1, -1], torch.int8, True),
    ])
    def forward(self, x, y):
        return torch.sub(x, y, alpha=2)


@register_test_case(module_factory=lambda: ElementwiseSubTensorInt8Module())
def ElementwiseSubTensorInt8Module_basic(module, tu: TestUtils):
        module.forward(
        tu.randint(3, 4, high=10).to(dtype=torch.int8),
        tu.randint(3, 4, high=10).to(dtype=torch.int8))


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
    module.forward(tu.randint(3, 4, high=10).to(dtype=torch.int32))


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
    module.forward(tu.randint(3, 4, high=10))


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
    module.forward(tu.randint(2, 3, high=10).to(torch.int32))


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


class ElementwiseAddScalar_NumToTensorFloat_Module(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        x = torch.ops.prim.NumToTensor(5.0)
        return torch.add(x, 3)


@register_test_case(
    module_factory=lambda: ElementwiseAddScalar_NumToTensorFloat_Module())
def ElementwiseAddScalar_NumToTensorFloat_Module_basic(module, tu: TestUtils):
    module.forward()


# ==============================================================================


class ElementwiseAddScalar_TensorLiteralInt32_Module(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.x = torch.tensor(2, dtype=torch.int32)

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        return torch.add(self.x, 3)


@register_test_case(
    module_factory=lambda: ElementwiseAddScalar_TensorLiteralInt32_Module())
def ElementwiseAddScalar_TensorLiteralInt32_Module_basic(module, tu: TestUtils):
    module.forward()


# ==============================================================================


class ElementwiseAddScalarInt8Module(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int8, True),
    ])
    def forward(self, x):
        return torch.add(x, 3, 2)


@register_test_case(module_factory=lambda: ElementwiseAddScalarInt8Module())
def ElementwiseAddScalarInt8Module_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, high=10).to(torch.int8))


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


class ElementwiseCloneChannelsLastMemoryFormatModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.clone(x, memory_format=torch.channels_last)


@register_test_case(
    module_factory=lambda: ElementwiseCloneChannelsLastMemoryFormatModule())
def ElementwiseCloneChannelsLastMemoryFormatModule_basic(
        module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4, 5))


# ==============================================================================


class LiftFreshCopyModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.lift_fresh_copy(x)


@register_test_case(module_factory=lambda: LiftFreshCopyModule())
def LiftFreshCopyModule_basic(module, tu: TestUtils):
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
    module.forward(tu.randint(3, 4, low=1, high=10).to(torch.int32))


# ==============================================================================


class ElementwiseExpm1Module(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.special.expm1(a)


@register_test_case(module_factory=lambda: ElementwiseExpm1Module())
def ElementwiseExpm1Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


# ==============================================================================


class ElementwiseExpm1IntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
    ])
    def forward(self, a):
        return torch.special.expm1(a)


@register_test_case(module_factory=lambda: ElementwiseExpm1IntModule())
def ElementwiseExpm1IntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, low=1, high=10).to(torch.int32))


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
    module.forward(tu.randint(3, 4, low=1, high=10).to(torch.int32))


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
    module.forward(tu.randint(3, 4, low=1, high=10).to(torch.int32))

# ==============================================================================


class ElementwiseAcosModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.acos(a)


@register_test_case(module_factory=lambda: ElementwiseAcosModule())
def ElementwiseAcosModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))

# ==============================================================================


class ElementwiseAcosIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
    ])
    def forward(self, a):
        return torch.acos(a)


@register_test_case(module_factory=lambda: ElementwiseAcosIntModule())
def ElementwiseAcosIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, low=1, high=10).to(torch.int32))

# ==============================================================================

class ElementwiseTanModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.tan(a)


@register_test_case(module_factory=lambda: ElementwiseTanModule())
def ElementwiseTanModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))

# ==============================================================================

class ElementwiseTanIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
    ])
    def forward(self, a):
        return torch.tan(a)


@register_test_case(module_factory=lambda: ElementwiseTanIntModule())
def ElementwiseTanIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, low=1, high=10).to(torch.int32))

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
    module.forward(tu.randint(2, 3, 4, 5, low=3, high=10), tu.randint(2, 3, 4, 5, low=10, high=100))

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
    module.forward(tu.rand(2, 3, 3, 5), tu.rand(2, 3, 3, 5))

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
    module.forward(torch.neg(tu.randint(2, 3, 4, 5, low=3, high=10)), torch.neg(tu.randint(2, 3, 4, 5, low=10, high=100)))

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
    module.forward(tu.randint(3, high=3), tu.randint(4, 3, high=3))


# ==============================================================================


class ElementwiseAtenLogicalOrOpPromoteBroadcastStaticShapeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([256], torch.float32, True),
        ([3, 256], torch.int64, True),
    ])
    def forward(self, x, y):
        return torch.ops.aten.logical_or(x, y)

@register_test_case(module_factory=lambda: ElementwiseAtenLogicalOrOpPromoteBroadcastStaticShapeModule())
def ElementwiseAtenLogicalOrOpPromoteBroadcastStaticShapeModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(256), tu.randint(3, 256, low=-1, high=2))


# ==============================================================================


class ElementwiseAtenLogicalAndOpModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.bool, True),
        ([-1, -1], torch.bool, True),
    ])
    def forward(self, x, y):
        return torch.ops.aten.logical_and(x, y)

@register_test_case(module_factory=lambda: ElementwiseAtenLogicalAndOpModule())
def ElementwiseAtenLogicalAndOpModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(4, 5, high=2).bool(), tu.randint(4, 5, high=2).bool())


# ==============================================================================


class ElementwiseAtenLogicalAndOpPromoteBroadcastModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.float32, True),
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, x, y):
        return torch.ops.aten.logical_and(x, y)

@register_test_case(module_factory=lambda: ElementwiseAtenLogicalAndOpPromoteBroadcastModule())
def ElementwiseAtenLogicalAndOpPromoteBroadcastModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(256), tu.randint(3, 256, low=-1, high=2))


# ==============================================================================


class ElementwiseAtenLogicalAndOpPromoteBroadcastStaticShapeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([256], torch.float32, True),
        ([3, 256], torch.int64, True),
    ])
    def forward(self, x, y):
        return torch.ops.aten.logical_and(x, y)

@register_test_case(module_factory=lambda: ElementwiseAtenLogicalAndOpPromoteBroadcastStaticShapeModule())
def ElementwiseAtenLogicalAndOpPromoteBroadcastStaticShapeModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(256), tu.randint(3, 256, low=-1, high=2))


# ==============================================================================


class ElementwiseAtenLogicalXorOpModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.bool, True),
        ([-1, -1], torch.bool, True),
    ])
    def forward(self, x, y):
        return torch.ops.aten.logical_xor(x, y)

@register_test_case(module_factory=lambda: ElementwiseAtenLogicalXorOpModule())
def ElementwiseAtenLogicalXorOpModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(4, 5, high=2).bool(), tu.randint(4, 5, high=2).bool())


# ==============================================================================


class ElementwiseAtenLogicalXorOpPromoteBroadcastModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.float32, True),
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, x, y):
        return torch.ops.aten.logical_xor(x, y)

@register_test_case(module_factory=lambda: ElementwiseAtenLogicalXorOpPromoteBroadcastModule())
def ElementwiseAtenLogicalXorOpPromoteBroadcastModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(256), tu.randint(3, 256, low=-1, high=2))


# ==============================================================================


class ElementwiseAtenLogicalXorOpPromoteBroadcastStaticShapeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([256], torch.float32, True),
        ([3, 256], torch.int64, True),
    ])
    def forward(self, x, y):
        return torch.ops.aten.logical_xor(x, y)

@register_test_case(module_factory=lambda: ElementwiseAtenLogicalXorOpPromoteBroadcastStaticShapeModule())
def ElementwiseAtenLogicalXorOpPromoteBroadcastStaticShapeModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(256), tu.randint(3, 256, low=-1, high=2))


# ==============================================================================


class ElementwiseAtenLogicalNotOpModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.bool, True),
    ])
    def forward(self, x):
        return torch.ops.aten.logical_not(x)

@register_test_case(module_factory=lambda: ElementwiseAtenLogicalNotOpModule())
def ElementwiseAtenLogicalNotOpModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(4, 5, high=2).bool())


# ==============================================================================

class ElementwiseAtenIsinfOpModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([2, 5], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.isinf(x)

@register_test_case(module_factory=lambda: ElementwiseAtenIsinfOpModule())
def ElementwiseAtenIsinfOpModule_basic(module, tu: TestUtils):
    test_input = torch.tensor(
        [
            [1, float('inf'), 2, float('-inf'), float('nan')],
            [1, float('inf'), float('-inf'), float('nan'), 3],
        ]
    )
    module.forward(test_input)


# ==============================================================================


class ElementwiseAtenIsneginfOpModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([2, 5], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.isneginf(x)

@register_test_case(module_factory=lambda: ElementwiseAtenIsneginfOpModule())
def ElementwiseAtenIsneginfOpModule_basic(module, tu:TestUtils):
    test_input = torch.tensor(
        [
            [1, float('-inf'), 2, float('inf'), float('nan')],
            [1, float('-inf'), float('inf'), float('nan'), 3],
        ]
    )
    module.forward(test_input)


# ==============================================================================


class ElementwiseAtenIsposinfOpModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([2, 5], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.isposinf(x)

@register_test_case(module_factory=lambda: ElementwiseAtenIsposinfOpModule())
def ElementwiseAtenIsposinfOpModule_basic(module, tu:TestUtils):
    test_input = torch.tensor(
        [
            [1, float('-inf'), 2, float('inf'), float('nan')],
            [1, float('-inf'), float('inf'), float('nan'), 3],
        ]
    )
    module.forward(test_input)


# ==============================================================================


class ElementwiseAtenLogicalNotOpPromoteModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, x):
        return torch.ops.aten.logical_not(x)

@register_test_case(module_factory=lambda: ElementwiseAtenLogicalNotOpPromoteModule())
def ElementwiseAtenLogicalNotOpPromoteModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(4, 5, low=-1, high=2))


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


class TriuModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([4,5], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.triu(x, 1)


@register_test_case(module_factory=lambda: TriuModule())
def TriuModule_basic(module, tu: TestUtils):
    x=torch.tensor([[ 0.5876, -0.0794, -1.8373,  0.6654, 0.2],
        [-0.2447,  0.9556, -1.2919,  1.3378, 0.3],
        [ 0.4333,  0.3146,  0.6576, -1.0432, 0.4],
        [-0.9888,  torch.nan, torch.inf, -torch.inf, 0.5]])
    module.forward(x)


# ==============================================================================


class TriuBroadcastModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([3,4,5,6], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.triu(x, 2)


@register_test_case(module_factory=lambda: TriuBroadcastModule())
def TriuBroadcastModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3,4,5,6))


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


# ==============================================================================


class AtenTrilModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.tril(x)


@register_test_case(module_factory=lambda: AtenTrilModule())
def AtenTrilModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(8, 8))


# ==============================================================================


class AtenTrilWithPosDiagonalModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.tril(x, diagonal=2)


@register_test_case(module_factory=lambda: AtenTrilWithPosDiagonalModule())
def AtenTrilWithPosDiagonalModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(9, 4, 3))


# ==============================================================================


class AtenTrilWithNegDiagonalModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.tril(x, diagonal=-4)


@register_test_case(module_factory=lambda: AtenTrilWithNegDiagonalModule())
def AtenTrilWithNegDiagonalModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 1, 5, 9))


# ==============================================================================


class AtenRoundFloatModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.round(x)


@register_test_case(module_factory=lambda: AtenRoundFloatModule())
def AtenRoundFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 5, low = -3.0, high = 3.0))


class AtenRoundFloatHalfToEvenModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.round(x)


@register_test_case(module_factory=lambda: AtenRoundFloatHalfToEvenModule())
def AtenRoundFloatHalfToEvenModule_basic(module, tu: TestUtils):
    module.forward(torch.FloatTensor([[0.5, 1.5], [-0.5, -1.5]]))


class AtenRoundIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, x):
        return torch.ops.aten.round(x)


@register_test_case(module_factory=lambda: AtenRoundIntModule())
def AtenRoundIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(5, 5, low = -10))


# ==============================================================================


class Fill_TensorFloat64WithFloat32(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, tensor):
        return torch.ops.aten.fill_(tensor, 3.0)


@register_test_case(module_factory=lambda: Fill_TensorFloat64WithFloat32())
def Fill_TensorFloat64WithFloat32_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 2, 4))


class Fill_TensorFloat64WithFloat32Static(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([3, 2, 4], torch.float32, True),
    ])
    def forward(self, tensor):
        return torch.ops.aten.fill_(tensor, 3.0)


@register_test_case(module_factory=lambda: Fill_TensorFloat64WithFloat32Static())
def Fill_TensorFloat64WithFloat32Static_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 2, 4))


class Fill_TensorFloat64WithFloat64(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float64, True),
    ])
    def forward(self, tensor):
        return torch.ops.aten.fill_(tensor, 3.0)


@register_test_case(module_factory=lambda: Fill_TensorFloat64WithFloat64())
def Fill_TensorFloat64WithFloat64_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 2, 4).to(torch.float64))


class Fill_TensorFloat64WithInt64(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float64, True),
    ])
    def forward(self, tensor):
        return torch.ops.aten.fill_(tensor, 3)


@register_test_case(module_factory=lambda: Fill_TensorFloat64WithInt64())
def Fill_TensorFloat64WithInt64_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 2, 4).to(torch.float64))


class Fill_TensorFloat64WithInt64Static(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([3, 2, 4], torch.float64, True),
    ])
    def forward(self, tensor):
        return torch.ops.aten.fill_(tensor, 3)


@register_test_case(module_factory=lambda: Fill_TensorFloat64WithInt64Static())
def Fill_TensorFloat64WithInt64Static_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 2, 4).to(torch.float64))


# ==============================================================================


class Fill_TensorFloat32WithFloat32(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([], torch.float32, True),
    ])
    def forward(self, tensor, value):
        return torch.ops.aten.fill_(tensor, value)

@register_test_case(module_factory=lambda: Fill_TensorFloat32WithFloat32())
def Fill_TensorFloat32WithFloat32_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 2, 4), tu.rand())


class Fill_TensorFloat32WithFloat64(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([], torch.float64, True),
    ])
    def forward(self, tensor, value):
        return torch.ops.aten.fill_(tensor, value)

@register_test_case(module_factory=lambda: Fill_TensorFloat32WithFloat64())
def Fill_TensorFloat32WithFloat64_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 2, 4), tu.rand().to(torch.float64))


class Fill_TensorFloat32WithInt64(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([], torch.int64, True),
    ])
    def forward(self, tensor, value):
        return torch.ops.aten.fill_(tensor, value)

@register_test_case(module_factory=lambda: Fill_TensorFloat32WithInt64())
def Fill_TensorFloat32WithInt64_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 2, 4), tu.randint())


# ==============================================================================


class TupleModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
    ])

    def forward(self, a, b):
        cond = True
        if cond:
            tuple = a, b
        else:
            tuple = a + b, None
        _, y = tuple
        return y


@register_test_case(module_factory=lambda: TupleModule())
def TupleModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 2), tu.rand(2, 2))


# ==============================================================================


class ElementwiseBitwiseRightShiftInt64Module(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, lhs, rhs):
        return torch.bitwise_right_shift(lhs, rhs)


@register_test_case(module_factory=lambda: ElementwiseBitwiseRightShiftInt64Module())
def ElementwiseBitwiseRightShiftInt64Module_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, low=-1000, high=1000), tu.randint(3, 4, low=0, high=64))


class ElementwiseBitwiseRightShiftInt32Module(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, 4], torch.int32, True),
        ([-1, 1], torch.int32, True),
    ])
    def forward(self, lhs, rhs):
        return torch.bitwise_right_shift(lhs, rhs)


@register_test_case(module_factory=lambda: ElementwiseBitwiseRightShiftInt32Module())
def ElementwiseBitwiseRightShiftInt32Module_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, low=-1000, high=1000).to(torch.int32), tu.randint(3, 1, low=0, high=32).to(torch.int32))


class ElementwiseBitwiseRightShiftInt8Module(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int8, True),
        ([-1, -1], torch.int8, True),
    ])
    def forward(self, lhs, rhs):
        return torch.bitwise_right_shift(lhs, rhs)


@register_test_case(module_factory=lambda: ElementwiseBitwiseRightShiftInt8Module())
def ElementwiseBitwiseRightShiftInt8Module_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, low=-100, high=100).to(torch.int8), tu.randint(3, 4, low=0, high=8).to(torch.int8))


# ==============================================================================


class ElementwiseBitwiseLeftShiftInt64Module(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, lhs, rhs):
        return torch.bitwise_left_shift(lhs, rhs)


@register_test_case(module_factory=lambda: ElementwiseBitwiseLeftShiftInt64Module())
def ElementwiseBitwiseLeftShiftInt64Module_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, low=-1000, high=1000), tu.randint(3, 4, low=0, high=64))


class ElementwiseBitwiseLeftShiftInt32Module(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, 4], torch.int32, True),
        ([-1, 1], torch.int32, True),
    ])
    def forward(self, lhs, rhs):
        return torch.bitwise_left_shift(lhs, rhs)


@register_test_case(module_factory=lambda: ElementwiseBitwiseLeftShiftInt32Module())
def ElementwiseBitwiseLeftShiftInt32Module_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, low=-1000, high=1000).to(torch.int32), tu.randint(3, 1, low=0, high=32).to(torch.int32))


class ElementwiseBitwiseLeftShiftInt8Module(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int8, True),
        ([-1, -1], torch.int8, True),
    ])
    def forward(self, lhs, rhs):
        return torch.bitwise_left_shift(lhs, rhs)


@register_test_case(module_factory=lambda: ElementwiseBitwiseLeftShiftInt8Module())
def ElementwiseBitwiseLeftShiftInt8Module_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, low=-100, high=100).to(torch.int8), tu.randint(3, 4, low=0, high=8).to(torch.int8))


# ==============================================================================


class ElementwiseBitwiseAndScalarInt64Module(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, x):
        return torch.bitwise_and(x, 15)


@register_test_case(module_factory=lambda: ElementwiseBitwiseAndScalarInt64Module())
def ElementwiseBitwiseAndScalarInt64Module_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, low=-1000, high=1000))


class ElementwiseBitwiseAndScalarInt32Module(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
    ])
    def forward(self, x):
        return torch.bitwise_and(x, 100)


@register_test_case(module_factory=lambda: ElementwiseBitwiseAndScalarInt32Module())
def ElementwiseBitwiseAndScalarInt32Module_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, low=-1000, high=1000).to(torch.int32))


class ElementwiseBitwiseAndScalarInt8Module(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int8, True),
    ])
    def forward(self, x):
        return torch.bitwise_and(x, 100)


@register_test_case(module_factory=lambda: ElementwiseBitwiseAndScalarInt8Module())
def ElementwiseBitwiseAndScalarInt8Module_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, low=-1000, high=1000).to(torch.int8))

# ==============================================================================

class ElementwiseQuantizePerTensorModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float, True),
    ])
    def forward(self, x):
        scale = 0.04
        zp = -110
        dtype = torch.qint8
        # We return the int representation as we can not map to quint8 type yet on boundaries.
        q = torch.quantize_per_tensor(x, scale, zp, dtype).int_repr()
        return q

@register_test_case(module_factory=lambda: ElementwiseQuantizePerTensorModule())
def ElementwiseQuantizePerTensorModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


# ==============================================================================

class ElementwiseDequantizePerTensorModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int8, True),
    ])
    def forward(self, x):
        qx = torch._make_per_tensor_quantized_tensor(x, 0.1, 8)
        qx = torch.dequantize(qx)
        return qx

@register_test_case(module_factory=lambda: ElementwiseDequantizePerTensorModule())
def ElementwiseDequantizePerTensorModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, low=-128, high=127).to(torch.int8))

# ==============================================================================

class ElementwiseDequantizePerChannelModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([3, 4], torch.int8, True),
        ([4], torch.int8, True),
        ([4], torch.float, True),
    ])
    def forward(self, x, zeropoint, scale):
        qx = torch._make_per_channel_quantized_tensor(x, scale, zeropoint, axis=1)
        qx = torch.dequantize(qx)
        return qx

@register_test_case(module_factory=lambda: ElementwiseDequantizePerChannelModule())
def ElementwiseDequantizePerChannelModule_basic(module, tu: TestUtils):
    module.forward(
        tu.randint(3, 4, low=-128, high=127).to(torch.int8),
        tu.randint(4, low=-128, high=127).to(torch.int8),
        tu.rand(4)
    )

# ==============================================================================

class GluStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([3, 24, 5], torch.float32, True)
    ])
    def forward(self, x):
        return torch.ops.aten.glu(x, dim=1)

@register_test_case(module_factory=lambda: GluStaticModule())
def  GluStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 24, 5))
