# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================


class AdaptiveAvgPool2dNonUnitOutputSizeStaticModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.aap2d = torch.nn.AdaptiveAvgPool2d((7, 7))

    @export
    @annotate_args([
        None,
        ([1, 512, 7, 7], torch.float32, True),
    ])
    def forward(self, x):
        return self.aap2d(x)


@register_test_case(
    module_factory=lambda: AdaptiveAvgPool2dNonUnitOutputSizeStaticModule())
def AdaptiveAvgPool2dNonUnitOutputSizeStaticModule_basic(
        module, tu: TestUtils):
    module.forward(tu.rand(1, 512, 7, 7))


class AdaptiveAvgPool2dNonUnitOutputSizeDynamicModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.aap2d = torch.nn.AdaptiveAvgPool2d((7, 7))

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return self.aap2d(x)


@register_test_case(
    module_factory=lambda: AdaptiveAvgPool2dNonUnitOutputSizeDynamicModule())
def AdaptiveAvgPool2dNonUnitOutputSizeDynamicModule_basic(
        module, tu: TestUtils):
    module.forward(tu.rand(1, 512, 7, 7))


class AdaptiveAvgPool2dUnitOutputSizeStaticModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.aap2d = torch.nn.AdaptiveAvgPool2d((1, 1))

    @export
    @annotate_args([
        None,
        ([1, 512, 7, 7], torch.float32, True),
    ])
    def forward(self, x):
        return self.aap2d(x)


@register_test_case(
    module_factory=lambda: AdaptiveAvgPool2dUnitOutputSizeStaticModule())
def AdaptiveAvgPool2dUnitOutputSizeStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 512, 7, 7))


class AdaptiveAvgPool2dUnitOutputSizeDynamicModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.aap2d = torch.nn.AdaptiveAvgPool2d((1, 1))

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return self.aap2d(x)


@register_test_case(
    module_factory=lambda: AdaptiveAvgPool2dUnitOutputSizeDynamicModule())
def AdaptiveAvgPool2dUnitOutputSizeDynamicModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 512, 7, 7))


# ==============================================================================


class MaxPool2dModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.mp2d = torch.nn.MaxPool2d(kernel_size=[6, 8],
                                       stride=[2, 2],
                                       padding=[3, 4],
                                       dilation=2)

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return self.mp2d(x)


@register_test_case(module_factory=lambda: MaxPool2dModule())
def MaxPool2dModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 20, 20, low=-1))


class MaxPool2dStaticModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.mp2d = torch.nn.MaxPool2d(kernel_size=[3, 3],
                                       stride=[2, 2],
                                       padding=[1, 1],
                                       dilation=[1, 1])

    @export
    @annotate_args([
        None,
        ([1, 64, 112, 112], torch.float32, True),
    ])
    def forward(self, x):
        return self.mp2d(x)


@register_test_case(module_factory=lambda: MaxPool2dStaticModule())
def MaxPool2dStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 64, 112, 112))


class MaxPool2dCeilModeTrueModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.mp2d = torch.nn.MaxPool2d(kernel_size=[6, 8],
                                       stride=[2, 2],
                                       padding=[3, 4],
                                       dilation=2,
                                       ceil_mode=True)

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return self.mp2d(x)


@register_test_case(module_factory=lambda: MaxPool2dCeilModeTrueModule())
def MaxPool2dCeilModeTrueModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 20, 20, low=0.5, high=1.0))


# ==============================================================================


class MaxPool2dWithIndicesModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.max_pool2d_with_indices(x,
                                                      kernel_size=[2, 2],
                                                      stride=[1, 1],
                                                      padding=[0, 0],
                                                      dilation=[1, 1])


@register_test_case(module_factory=lambda: MaxPool2dWithIndicesModule())
def MaxPool2dWithIndicesModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 8, 8, low=0.5, high=1.0))


class MaxPool2dWithIndicesFullSizeKernelModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.max_pool2d_with_indices(x,
                                                      kernel_size=[4, 4],
                                                      stride=1,
                                                      padding=0,
                                                      dilation=1)


@register_test_case(
    module_factory=lambda: MaxPool2dWithIndicesFullSizeKernelModule())
def MaxPool2dWithIndicesFullSizeKernelModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4, 4, low=0.5, high=1.0))


class MaxPool2dWithIndicesNonDefaultPaddingModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.max_pool2d_with_indices(x,
                                                      kernel_size=[4, 8],
                                                      stride=[1, 1],
                                                      padding=[2, 4],
                                                      dilation=1)


@register_test_case(
    module_factory=lambda: MaxPool2dWithIndicesNonDefaultPaddingModule())
def MaxPool2dWithIndicesNonDefaultPaddingModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4, 16, 16, low=-1.5, high=1.0))


class MaxPool2dWithIndicesNonDefaultStrideModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.max_pool2d_with_indices(x,
                                                      kernel_size=[4, 4],
                                                      stride=[1, 2],
                                                      padding=0,
                                                      dilation=1)


@register_test_case(
    module_factory=lambda: MaxPool2dWithIndicesNonDefaultStrideModule())
def MaxPool2dWithIndicesNonDefaultStrideModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 4, 16, 80, low=0.5, high=2.0))


class MaxPool2dWithIndicesNonDefaultDilationModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.max_pool2d_with_indices(x,
                                                      kernel_size=[4, 4],
                                                      stride=[1, 1],
                                                      padding=0,
                                                      dilation=[2, 2])


@register_test_case(
    module_factory=lambda: MaxPool2dWithIndicesNonDefaultDilationModule())
def MaxPool2dWithIndicesNonDefaultDilationModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 4, 16, 80, low=0.5, high=2.0))


class MaxPool2dWithIndicesNonDefaultParamsModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.max_pool2d_with_indices(x,
                                                      kernel_size=[8, 4],
                                                      stride=[2, 2],
                                                      padding=[1, 2],
                                                      dilation=[2, 2])


@register_test_case(
    module_factory=lambda: MaxPool2dWithIndicesNonDefaultParamsModule())
def MaxPool2dWithIndicesNonDefaultParamsModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 4, 16, 80, low=-0.5, high=4.0))


class MaxPool2dWithIndicesAllNegativeValuesModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.max_pool2d_with_indices(x,
                                                      kernel_size=[4, 8],
                                                      stride=[1, 1],
                                                      padding=[2, 4],
                                                      dilation=1)


@register_test_case(
    module_factory=lambda: MaxPool2dWithIndicesAllNegativeValuesModule())
def MaxPool2dWithIndicesAllNegativeValuesModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4, 16, 16, low=-4.5, high=-1.0))


class MaxPool2dWithIndicesStaticModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([2, 4, 16, 16], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.max_pool2d_with_indices(x,
                                                      kernel_size=[4, 8],
                                                      stride=[1, 1],
                                                      padding=[2, 4],
                                                      dilation=1)


@register_test_case(module_factory=lambda: MaxPool2dWithIndicesStaticModule())
def MaxPool2dWithIndicesStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4, 16, 16, low=-4.5, high=-1.0))


class MaxPool2dWithIndicesAllOnesModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.max_pool2d_with_indices(x,
                                                      kernel_size=[2, 2],
                                                      stride=[1, 1],
                                                      padding=[0, 0],
                                                      dilation=[1, 1])


@register_test_case(module_factory=lambda: MaxPool2dWithIndicesAllOnesModule())
def MaxPool2dWithIndicesAllOnesModule_basic(module, tu: TestUtils):
    module.forward(torch.ones(1, 1, 8, 8))


class MaxPool2dWithIndicesCeilModeTrueModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.max_pool2d_with_indices(x,
                                                      kernel_size=[2, 2],
                                                      stride=[1, 1],
                                                      padding=[0, 0],
                                                      dilation=[1, 1],
                                                      ceil_mode=True)


@register_test_case(
    module_factory=lambda: MaxPool2dWithIndicesCeilModeTrueModule())
def MaxPool2dWithIndicesCeilModeTrueModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 8, 8, low=0.5, high=1.0))


# ==============================================================================


class MaxPool2dWithIndicesBackwardStatic4DModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([2, 4, 7, 6], torch.float32, True),
        ([2, 4, 6, 5], torch.float32, True),
        ([2, 4, 7, 6], torch.int64, True),
    ])
    def forward(self, output, input, indices):
        kernel_size = [2, 2]
        stride = [1, 1]
        padding = [1, 1]
        dilation = [1, 1]
        ceil_mode = False
        return torch.ops.aten.max_pool2d_with_indices_backward(
            output, input, kernel_size, stride, padding, dilation, ceil_mode,
            indices)


@register_test_case(
    module_factory=lambda: MaxPool2dWithIndicesBackwardStatic4DModule())
def MaxPool2dWithIndicesBackwardStatic4DModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4, 7, 6), tu.rand(2, 4, 6, 5),
                   tu.randint(2, 4, 7, 6, high=16))


class MaxPool2dWithIndicesBackwardStatic3DModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([4, 7, 6], torch.float32, True),
        ([4, 6, 5], torch.float32, True),
        ([4, 7, 6], torch.int64, True),
    ])
    def forward(self, output, input, indices):
        kernel_size = [2, 2]
        stride = [1, 1]
        padding = [1, 1]
        dilation = [1, 1]
        ceil_mode = False
        return torch.ops.aten.max_pool2d_with_indices_backward(
            output, input, kernel_size, stride, padding, dilation, ceil_mode,
            indices)


@register_test_case(
    module_factory=lambda: MaxPool2dWithIndicesBackwardStatic3DModule())
def MaxPool2dWithIndicesBackwardStatic3DModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 7, 6), tu.rand(4, 6, 5),
                   tu.randint(4, 7, 6, high=16))


class MaxPool2dWithIndicesBackwardDynamic4DModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
        ([-1, -1, -1, -1], torch.float32, True),
        ([-1, -1, -1, -1], torch.int64, True),
    ])
    def forward(self, output, input, indices):
        kernel_size = [2, 2]
        stride = [1, 1]
        padding = [1, 1]
        dilation = [1, 1]
        ceil_mode = False
        return torch.ops.aten.max_pool2d_with_indices_backward(
            output, input, kernel_size, stride, padding, dilation, ceil_mode,
            indices)


@register_test_case(
    module_factory=lambda: MaxPool2dWithIndicesBackwardDynamic4DModule())
def MaxPool2dWithIndicesBackwardDynamic4DModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4, 7, 6), tu.rand(2, 4, 6, 5),
                   tu.randint(2, 4, 7, 6, high=16))


class MaxPool2dWithIndicesBackwardDynamic3DModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.int64, True),
    ])
    def forward(self, output, input, indices):
        kernel_size = [2, 2]
        stride = [1, 1]
        padding = [1, 1]
        dilation = [1, 1]
        ceil_mode = False
        return torch.ops.aten.max_pool2d_with_indices_backward(
            output, input, kernel_size, stride, padding, dilation, ceil_mode,
            indices)


@register_test_case(
    module_factory=lambda: MaxPool2dWithIndicesBackwardDynamic3DModule())
def MaxPool2dWithIndicesBackwardDynamic3DModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 7, 6), tu.rand(2, 6, 5),
                   tu.randint(2, 7, 6, high=16))


# ==============================================================================


class AvgPool2dFloatModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.ap2d = torch.nn.AvgPool2d(kernel_size=[6, 8],
                                       stride=[2, 2],
                                       padding=[3, 4],
                                       ceil_mode=False,
                                       count_include_pad=True,
                                       divisor_override=None)

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return self.ap2d(x)

@register_test_case(module_factory=lambda: AvgPool2dFloatModule())
def AvgPool2dFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4, 20, 20, low=-1))

class AvgPool2dIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.ap2d = torch.nn.AvgPool2d(kernel_size=[6, 8],
                                       stride=[2, 2],
                                       padding=[3, 4],
                                       ceil_mode=False,
                                       count_include_pad=True,
                                       divisor_override=None)

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.int64, True),
    ])
    def forward(self, x):
        return self.ap2d(x)


@register_test_case(module_factory=lambda: AvgPool2dIntModule())
def AvgPool2dIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(2, 4, 20, 20, high=100))


class AvgPool2dStaticModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.ap2d = torch.nn.AvgPool2d(kernel_size=[6, 8],
                                       stride=[2, 2],
                                       padding=[3, 4],
                                       ceil_mode=False,
                                       count_include_pad=True,
                                       divisor_override=None)

    @export
    @annotate_args([
        None,
        ([2, 2, 10, 20], torch.float32, True),
    ])
    def forward(self, x):
        return self.ap2d(x)


@register_test_case(module_factory=lambda: AvgPool2dStaticModule())
def AvgPool2dStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 2, 10, 20, low=-1))


class AvgPool2dDivisorOverrideModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.ap2d = torch.nn.AvgPool2d(kernel_size=[4, 8],
                                       stride=[2, 3],
                                       padding=[2, 4],
                                       ceil_mode=False,
                                       count_include_pad=True,
                                       divisor_override=22)

    @export
    @annotate_args([
        None,
        ([4, 4, 20, 20], torch.float32, True),
    ])
    def forward(self, x):
        return self.ap2d(x)


@register_test_case(module_factory=lambda: AvgPool2dDivisorOverrideModule())
def AvgPool2dDivisorOverrideModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 4, 20, 20, low=-1))


class AvgPool2dCeilModeTrueModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.ap2d = torch.nn.AvgPool2d(kernel_size=[6, 8],
                                       stride=[2, 2],
                                       padding=[3, 4],
                                       ceil_mode=False,
                                       count_include_pad=True,
                                       divisor_override=None)

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return self.ap2d(x)

@register_test_case(module_factory=lambda: AvgPool2dCeilModeTrueModule())
def AvgPool2dCeilModeTrueModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4, 20, 20, low=0.5, high=1.0))
