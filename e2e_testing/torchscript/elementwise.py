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


@register_test_case(
    module_factory=lambda: ElementwiseUnsqueezeNegDimsModule())
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
    module.forward(2*tu.rand(5, 3) - 0.5)


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
        return torch.minimum(x, y)


@register_test_case(module_factory=lambda: ElementwiseMinimumModule())
def ElementwiseMinimumModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5), tu.rand(3, 5))
    module.forward(tu.nans(3, 5), tu.rand(3, 5))


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
        return torch.maximum(x, y)


@register_test_case(module_factory=lambda: ElementwiseMaximumModule())
def ElementwiseMaximumModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5), tu.rand(3, 5))
    module.forward(tu.nans(3, 5), tu.rand(3, 5))

# ==============================================================================


class ElementwiseGtScalarModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.gt(x, 0.6)


@register_test_case(module_factory=lambda: ElementwiseGtScalarModule())
def ElementwiseGtScalarModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5))

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
        # TODO: It would be great to return all of these, so they get checked
        # individually, but RefBackend doesn't support multiple returns.
        # Instead, multiply them together, which has some chance of propagating
        # all the values.
        float_min = torch.clamp(x, min=-2.0)
        int_min = torch.clamp(x, min=-3)
        float_max = torch.clamp(x, max=2.0)
        int_max = torch.clamp(x, max=3)
        both = torch.clamp(x, min=-5, max=5)
        return float_min * int_min * float_max * int_max * both


@register_test_case(module_factory=lambda: ElementwiseClampModule())
def ElementwiseClampModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5, low=-10, high=10))

# ==============================================================================
class RsubModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.rsub(x, 3.0, alpha=1.0)

@register_test_case(module_factory=lambda: RsubModule())
def RsubModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))

class RsubModule_noalpha(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.rsub(x, 2.0)

@register_test_case(module_factory=lambda: RsubModule_noalpha())
def RsubModule_noalpha_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))
    
# ==============================================================================


class ElementwiseMulScalarModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.mul(x, 100.0)

@register_test_case(module_factory=lambda: ElementwiseMulScalarModule())
def ElementwiseMulScalarModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))



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


@register_test_case(
    module_factory=lambda: ElementwiseMulTensorFloatModule())
def ElementwiseMulTensorFloatModule_basic(module, tu: TestUtils):
    module.forward(
        tu.rand(4),
        tu.rand(4).type(torch.float64))

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


@register_test_case(
    module_factory=lambda: ElementwiseMulTensorIntModule())
def ElementwiseMulTensorIntModule_basic(module, tu: TestUtils):
    module.forward(
      torch.randint(10, [4]).type(torch.int32),
      torch.randint(10, [4]))


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
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True)
    ])
    def forward(self, x):
        return x.to(torch.int64)

@register_test_case(module_factory=lambda: ElementwiseToDtypeF32ToI64Module())
def ElementwiseToDtypeF32ToI64Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5))

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


@register_test_case(
    module_factory=lambda: ElementwiseDivTensorFloatModule())
def ElementwiseDivTensorFloatModule_basic(module, tu: TestUtils):
    module.forward(
        tu.rand(4),
        tu.rand(4).type(torch.float64))


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
    module.forward(torch.randint(-10, 10, (3, 4)).to(torch.int32),
                   torch.randint(-10, 10, (3, 4)))


