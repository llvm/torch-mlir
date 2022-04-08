# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch
from torch_mlir_e2e_test.torchscript.framework import TestUtils
from torch_mlir_e2e_test.torchscript.registry import register_test_case
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export

# ==============================================================================


class Conv2dNoPaddingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(2, 10, 3, bias=False)
        self.train(False)

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return self.conv(x)


@register_test_case(module_factory=lambda: Conv2dNoPaddingModule())
def Conv2dNoPaddingModule_basic(module, tu: TestUtils):
    t = tu.rand(5, 2, 10, 20)
    module.forward(t)


class Conv2dBiasNoPaddingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(2, 10, 3, bias=True)
        self.train(False)

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return self.conv(x)


@register_test_case(module_factory=lambda: Conv2dBiasNoPaddingModule())
def Conv2dBiasNoPaddingModule_basic(module, tu: TestUtils):
    t = tu.rand(5, 2, 10, 20)
    module.forward(t)


class Conv2dWithPaddingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(2, 10, 3, bias=False, padding=3)
        self.train(False)

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return self.conv(x)


@register_test_case(module_factory=lambda: Conv2dWithPaddingModule())
def Conv2dWithPaddingModule_basic(module, tu: TestUtils):
    t = tu.rand(5, 2, 10, 20)
    module.forward(t)


class Conv2dWithPaddingDilationStrideModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(in_channels=2,
                                    out_channels=10,
                                    kernel_size=3,
                                    padding=3,
                                    stride=2,
                                    dilation=3,
                                    bias=False)
        self.train(False)

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return self.conv(x)


@register_test_case(
    module_factory=lambda: Conv2dWithPaddingDilationStrideModule())
def Conv2dWithPaddingDilationStrideModule_basic(module, tu: TestUtils):
    t = tu.rand(5, 2, 10, 20)
    module.forward(t)


class Conv2dWithPaddingDilationStrideStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(in_channels=2,
                                    out_channels=10,
                                    kernel_size=3,
                                    padding=3,
                                    stride=2,
                                    dilation=3,
                                    bias=False)
        self.train(False)

    @export
    @annotate_args([
        None,
        ([5, 2, 10, 20], torch.float32, True),
    ])
    def forward(self, x):
        return self.conv(x)


@register_test_case(
    module_factory=lambda: Conv2dWithPaddingDilationStrideStaticModule())
def Conv2dWithPaddingDilationStrideStaticModule_basic(module, tu: TestUtils):
    t = tu.rand(5, 2, 10, 20)
    module.forward(t)

# ==============================================================================

class ConvolutionModule1D(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, inputVec, weight):
        return torch.ops.aten.convolution(inputVec,
                                           weight,
                                           bias=None,
                                           stride=[1],
                                           padding=[0],
                                           dilation=[1],
                                           transposed=False,
                                           output_padding=[0],
                                           groups=1)

@register_test_case(module_factory=lambda: ConvolutionModule1D())
def ConvolutionModule1D_basic(module, tu: TestUtils):
    module.forward(torch.randn(3, 3, 10), torch.randn(3, 3, 2))

class ConvolutionModule2D(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, inputVec, weight):
        return torch.ops.aten.convolution(inputVec,
                                          weight,
                                          bias=None,
                                          stride=[1, 1],
                                          padding=[0, 0],
                                          dilation=[1, 1],
                                          transposed=False,
                                          output_padding=[0, 0],
                                          groups=1)

@register_test_case(module_factory=lambda: ConvolutionModule2D())
def ConvolutionModule2D_basic(module, tu: TestUtils):
    module.forward(torch.randn(3, 3, 10, 10), torch.randn(3, 3, 2, 2))

class ConvolutionModule3D(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1, -1], torch.float32, True),
        ([-1, -1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, inputVec, weight):
        return torch.ops.aten.convolution(inputVec,
                                           weight,
                                           bias=None,
                                           stride=[1, 1, 1],
                                           padding=[0, 0, 0],
                                           dilation=[1, 1, 1],
                                           transposed=False,
                                           output_padding=[0, 0, 0],
                                           groups=1)

@register_test_case(module_factory=lambda: ConvolutionModule3D())
def ConvolutionModule3D_basic(module, tu: TestUtils):
    module.forward(torch.randn(3, 3, 10, 10, 10), torch.randn(3, 3, 2, 2, 2))

class ConvolutionModule2DStatic(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([3, 3, 10, 10], torch.float32, True),
        ([3, 3, 2, 2], torch.float32, True),
    ])
    def forward(self, inputVec, weight):
        return torch.ops.aten.convolution(inputVec,
                                          weight,
                                          bias=None,
                                          stride=[1, 1],
                                          padding=[0, 0],
                                          dilation=[1, 1],
                                          transposed=False,
                                          output_padding=[0, 0],
                                          groups=1)

@register_test_case(module_factory=lambda: ConvolutionModule2DStatic())
def ConvolutionModule2DStatic_basic(module, tu: TestUtils):
    module.forward(torch.randn(3, 3, 10, 10), torch.randn(3, 3, 2, 2))

class ConvolutionModule2DStrided(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, inputVec, weight):
        return torch.ops.aten.convolution(inputVec,
                                          weight,
                                          bias=None,
                                          stride=[3, 3],
                                          padding=[2, 2],
                                          dilation=[1, 1],
                                          transposed=False,
                                          output_padding=[0, 0],
                                          groups=1)

@register_test_case(module_factory=lambda: ConvolutionModule2DStrided())
def ConvolutionModule2DStrided_basic(module, tu: TestUtils):
    module.forward(torch.randn(3, 3, 10, 10), torch.randn(3, 3, 2, 2))
