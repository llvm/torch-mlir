# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch
from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

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

class Convolution1DModule(torch.nn.Module):
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

@register_test_case(module_factory=lambda: Convolution1DModule())
def Convolution1DModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(3, 3, 10), torch.randn(3, 3, 2))

class Convolution2DModule(torch.nn.Module):
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

@register_test_case(module_factory=lambda: Convolution2DModule())
def Convolution2DModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(3, 3, 10, 10), torch.randn(3, 3, 2, 2))

class Convolution3DModule(torch.nn.Module):
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

@register_test_case(module_factory=lambda: Convolution3DModule())
def Convolution3DModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(3, 3, 10, 10, 10), torch.randn(3, 3, 2, 2, 2))

class Convolution2DStaticModule(torch.nn.Module):
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

@register_test_case(module_factory=lambda: Convolution2DStaticModule())
def Convolution2DStaticModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(3, 3, 10, 10), torch.randn(3, 3, 2, 2))

class Convolution2DStridedModule(torch.nn.Module):
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

@register_test_case(module_factory=lambda: Convolution2DStridedModule())
def Convolution2DStridedModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(3, 3, 10, 10), torch.randn(3, 3, 2, 2))

class _Convolution2DAllFalseModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, inputVec, weight):
        return torch.ops.aten._convolution(inputVec,
                                          weight,
                                          bias=None,
                                          stride=[3, 3],
                                          padding=[2, 2],
                                          dilation=[1, 1],
                                          transposed=False,
                                          output_padding=[0, 0],
                                          groups=1,
                                          benchmark=False,
                                          deterministic=False,
                                          cudnn_enabled=False,
                                          allow_tf32=False)

@register_test_case(module_factory=lambda: _Convolution2DAllFalseModule())
def _Convolution2DAllFalseModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(3, 3, 10, 10), torch.randn(3, 3, 2, 2))

class _Convolution2DBenchmarkModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, inputVec, weight):
        return torch.ops.aten._convolution(inputVec,
                                          weight,
                                          bias=None,
                                          stride=[3, 3],
                                          padding=[2, 2],
                                          dilation=[1, 1],
                                          transposed=False,
                                          output_padding=[0, 0],
                                          groups=1,
                                          benchmark=True,
                                          deterministic=False,
                                          cudnn_enabled=False,
                                          allow_tf32=False)

@register_test_case(module_factory=lambda: _Convolution2DBenchmarkModule())
def _Convolution2DBenchmarkModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(3, 3, 10, 10), torch.randn(3, 3, 2, 2))

class _Convolution2DDeterministicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, inputVec, weight):
        return torch.ops.aten._convolution(inputVec,
                                          weight,
                                          bias=None,
                                          stride=[3, 3],
                                          padding=[2, 2],
                                          dilation=[1, 1],
                                          transposed=False,
                                          output_padding=[0, 0],
                                          groups=1,
                                          benchmark=False,
                                          deterministic=True,
                                          cudnn_enabled=False,
                                          allow_tf32=False)

@register_test_case(module_factory=lambda: _Convolution2DDeterministicModule())
def _Convolution2DDeterministicModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(3, 3, 10, 10), torch.randn(3, 3, 2, 2))

class _Convolution2DCudnnModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, inputVec, weight):
        return torch.ops.aten._convolution(inputVec,
                                          weight,
                                          bias=None,
                                          stride=[3, 3],
                                          padding=[2, 2],
                                          dilation=[1, 1],
                                          transposed=False,
                                          output_padding=[0, 0],
                                          groups=1,
                                          benchmark=False,
                                          deterministic=False,
                                          cudnn_enabled=True,
                                          allow_tf32=False)

@register_test_case(module_factory=lambda: _Convolution2DCudnnModule())
def _Convolution2DCudnnModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(3, 3, 10, 10), torch.randn(3, 3, 2, 2))

class _Convolution2DTF32Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, inputVec, weight):
        return torch.ops.aten._convolution(inputVec,
                                          weight,
                                          bias=None,
                                          stride=[3, 3],
                                          padding=[2, 2],
                                          dilation=[1, 1],
                                          transposed=False,
                                          output_padding=[0, 0],
                                          groups=1,
                                          benchmark=False,
                                          deterministic=False,
                                          cudnn_enabled=False,
                                          allow_tf32=True)

@register_test_case(module_factory=lambda: _Convolution2DTF32Module())
def _Convolution2DTF32Module_basic(module, tu: TestUtils):
    module.forward(torch.randn(3, 3, 10, 10), torch.randn(3, 3, 2, 2))

class _ConvolutionDeprecated2DAllFalseModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, inputVec, weight):
        return torch.ops.aten._convolution(inputVec,
                                           weight,
                                           bias=None,
                                           stride=[3, 3],
                                           padding=[2, 2],
                                           dilation=[1, 1],
                                           transposed=False,
                                           output_padding=[0, 0],
                                           groups=1,
                                           benchmark=False,
                                           deterministic=False,
                                           cudnn_enabled=False)

@register_test_case(module_factory=lambda: _ConvolutionDeprecated2DAllFalseModule())
def _ConvolutionDeprecated2DAllFalseModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(3, 3, 10, 10), torch.randn(3, 3, 2, 2))

class _ConvolutionDeprecated2DBenchmarkModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, inputVec, weight):
        return torch.ops.aten._convolution(inputVec,
                                           weight,
                                           bias=None,
                                           stride=[3, 3],
                                           padding=[2, 2],
                                           dilation=[1, 1],
                                           transposed=False,
                                           output_padding=[0, 0],
                                           groups=1,
                                           benchmark=True,
                                           deterministic=False,
                                           cudnn_enabled=False)

@register_test_case(module_factory=lambda: _ConvolutionDeprecated2DBenchmarkModule())
def _ConvolutionDeprecated2DBenchmarkModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(3, 3, 10, 10), torch.randn(3, 3, 2, 2))

class _ConvolutionDeprecated2DDeterministicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, inputVec, weight):
        return torch.ops.aten._convolution(inputVec,
                                           weight,
                                           bias=None,
                                           stride=[3, 3],
                                           padding=[2, 2],
                                           dilation=[1, 1],
                                           transposed=False,
                                           output_padding=[0, 0],
                                           groups=1,
                                           benchmark=False,
                                           deterministic=True,
                                           cudnn_enabled=False)

@register_test_case(module_factory=lambda: _ConvolutionDeprecated2DDeterministicModule())
def _ConvolutionDeprecated2DDeterministicModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(3, 3, 10, 10), torch.randn(3, 3, 2, 2))

class _ConvolutionDeprecated2DCudnnModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, inputVec, weight):
        return torch.ops.aten._convolution(inputVec,
                                           weight,
                                           bias=None,
                                           stride=[3, 3],
                                           padding=[2, 2],
                                           dilation=[1, 1],
                                           transposed=False,
                                           output_padding=[0, 0],
                                           groups=1,
                                           benchmark=False,
                                           deterministic=False,
                                           cudnn_enabled=True)

@register_test_case(module_factory=lambda: _ConvolutionDeprecated2DCudnnModule())
def _ConvolutionDeprecated2DCudnnModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(3, 3, 10, 10), torch.randn(3, 3, 2, 2))

class ConvolutionModule2DGroups(torch.nn.Module):
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
                                          groups=4)

@register_test_case(module_factory=lambda: ConvolutionModule2DGroups())
def ConvolutionModule2DGroups_basic(module, tu: TestUtils):
    module.forward(torch.randn(1, 32, 4, 4), torch.randn(32, 8, 3, 3))

# ==============================================================================

class ConvolutionModule2DTranspose(torch.nn.Module):

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
                                          padding=[1, 1],
                                          dilation=[1, 1],
                                          transposed=True,
                                          output_padding=[0, 0],
                                          groups=1)


@register_test_case(module_factory=lambda: ConvolutionModule2DTranspose())
def ConvolutionModule2DTranspose_basic(module, tu: TestUtils):
    module.forward(torch.randn(3, 3, 4, 4), torch.randn(3, 3, 2, 2))

class ConvolutionModule2DTransposeStrided(torch.nn.Module):

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
                                          stride=[2, 2],
                                          padding=[1, 1],
                                          dilation=[1, 1],
                                          transposed=True,
                                          output_padding=[0, 0],
                                          groups=1)


@register_test_case(module_factory=lambda: ConvolutionModule2DTransposeStrided())
def ConvolutionModule2DTransposeStrided_basic(module, tu: TestUtils):
    module.forward(torch.randn(5, 2, 5, 6), torch.randn(2, 5, 2, 2))

class ConvolutionModule2DTransposeStridedStatic(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([5, 2, 5, 6], torch.float32, True),
        ([2, 5, 2, 2], torch.float32, True),
    ])
    def forward(self, inputVec, weight):
        return torch.ops.aten.convolution(inputVec,
                                          weight,
                                          bias=None,
                                          stride=[2, 2],
                                          padding=[1, 1],
                                          dilation=[1, 1],
                                          transposed=True,
                                          output_padding=[0, 0],
                                          groups=1)


@register_test_case(module_factory=lambda: ConvolutionModule2DTransposeStridedStatic())
def ConvolutionModule2DTransposeStridedStatic_basic(module, tu: TestUtils):
    module.forward(torch.randn(5, 2, 5, 6), torch.randn(2, 5, 2, 2))


class Conv_Transpose1dModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, inputVec, weight):
        return torch.ops.aten.conv_transpose1d(inputVec,
                                               weight,
                                               bias=None,
                                               stride=[2],
                                               padding=[1],
                                               dilation=[1],
                                               output_padding=[0],
                                               groups=1)


@register_test_case(module_factory=lambda: Conv_Transpose1dModule())
def Conv_Transpose1dModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(5, 2, 5), torch.randn(2, 5, 2))


class Conv_Transpose2dModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, inputVec, weight):
        return torch.ops.aten.conv_transpose2d(inputVec,
                                               weight,
                                               bias=None,
                                               stride=[2, 2],
                                               padding=[1, 1],
                                               dilation=[1, 1],
                                               output_padding=[0, 0],
                                               groups=1)


@register_test_case(module_factory=lambda: Conv_Transpose2dModule())
def Conv_Transpose2dModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(5, 2, 5, 6), torch.randn(2, 5, 2, 2))

class Conv_Transpose3dModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1, -1], torch.float32, True),
        ([-1, -1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, inputVec, weight):
        return torch.ops.aten.conv_transpose3d(inputVec,
                                               weight,
                                               bias=None,
                                               stride=[2, 2, 2],
                                               padding=[1, 1, 1],
                                               dilation=[1, 1, 1],
                                               output_padding=[0, 0, 0],
                                               groups=1)


@register_test_case(module_factory=lambda: Conv_Transpose3dModule())
def Conv_Transpose3dModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(5, 2, 5, 6, 4), torch.randn(2, 5, 2, 2, 2))
