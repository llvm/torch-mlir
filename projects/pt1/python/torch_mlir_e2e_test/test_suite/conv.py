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

    def __init__(self, out_channels, groups):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(in_channels=4,
                                    out_channels=out_channels,
                                    kernel_size=3,
                                    padding=3,
                                    stride=2,
                                    dilation=3,
                                    bias=False,
                                    groups=groups)
        self.train(False)

    @export
    @annotate_args([
        None,
        ([5, 4, 10, 20], torch.float32, True),
    ])
    def forward(self, x):
        return self.conv(x)


@register_test_case(
    module_factory=lambda: Conv2dWithPaddingDilationStrideStaticModule(out_channels=10, groups=1))
def Conv2dWithPaddingDilationStrideStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 4, 10, 20))


@register_test_case(
    module_factory=lambda: Conv2dWithPaddingDilationStrideStaticModule(out_channels=4, groups=4))
def Conv2dWithPaddingDilationStrideStaticModule_depthwise(module, tu: TestUtils):
    module.forward(tu.rand(5, 4, 10, 20))


@register_test_case(
    module_factory=lambda: Conv2dWithPaddingDilationStrideStaticModule(out_channels=8, groups=4))
def Conv2dWithPaddingDilationStrideStaticModule_depthwise_multiplier(module, tu: TestUtils):
    module.forward(tu.rand(5, 4, 10, 20))


@register_test_case(
    module_factory=lambda: Conv2dWithPaddingDilationStrideStaticModule(out_channels=4, groups=2))
def Conv2dWithPaddingDilationStrideStaticModule_grouped(module, tu: TestUtils):
    module.forward(tu.rand(5, 4, 10, 20))


@register_test_case(
    module_factory=lambda: Conv2dWithPaddingDilationStrideStaticModule(out_channels=8, groups=2))
def Conv2dWithPaddingDilationStrideStaticModule_grouped_multiplier(module, tu: TestUtils):
    module.forward(tu.rand(5, 4, 10, 20))


# ==============================================================================

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
    module.forward(tu.rand(3, 3, 10, 10), tu.rand(3, 3, 2, 2))


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
    module.forward(tu.rand(3, 3, 10, 10), tu.rand(3, 3, 2, 2))

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
    module.forward(tu.rand(3, 3, 10, 10), tu.rand(3, 3, 2, 2))

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
    module.forward(tu.rand(3, 3, 10, 10), tu.rand(3, 3, 2, 2))

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
    module.forward(tu.rand(3, 3, 10, 10), tu.rand(3, 3, 2, 2))

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
    module.forward(tu.rand(3, 3, 10, 10), tu.rand(3, 3, 2, 2))

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
    module.forward(tu.rand(3, 3, 10, 10), tu.rand(3, 3, 2, 2))

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
    module.forward(tu.rand(3, 3, 10, 10), tu.rand(3, 3, 2, 2))

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
    module.forward(tu.rand(3, 3, 10, 10), tu.rand(3, 3, 2, 2))

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
    module.forward(tu.rand(3, 3, 10, 10), tu.rand(3, 3, 2, 2))

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
    module.forward(tu.rand(3, 3, 10, 10), tu.rand(3, 3, 2, 2))

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
    module.forward(tu.rand(3, 3, 10, 10), tu.rand(3, 3, 2, 2))

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
    module.forward(tu.rand(1, 32, 4, 4), tu.rand(32, 8, 3, 3))

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
    module.forward(tu.rand(3, 3, 4, 4), tu.rand(3, 3, 2, 2))

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
    module.forward(tu.rand(5, 2, 5, 6), tu.rand(2, 5, 2, 2))

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
    module.forward(tu.rand(5, 2, 5, 6), tu.rand(2, 5, 2, 2))

class ConvolutionModule2DTransposeNonUnitOutputPadding(torch.nn.Module):
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
                                          output_padding=[1, 1],
                                          groups=1)

@register_test_case(module_factory=lambda: ConvolutionModule2DTransposeNonUnitOutputPadding())
def ConvolutionModule2DTransposeNonUnitOutputPadding_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 2, 4, 4), tu.rand(2, 2, 3, 3))


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
    module.forward(tu.rand(5, 2, 5, 6), tu.rand(2, 5, 2, 2))


class UpSampleNearest2d(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float64, True),
    ])
    def forward(self, input):
        return torch.ops.aten.upsample_nearest2d(input,
                                               output_size=[18, 48],
                                               scales_h=3.0,
                                               scales_w=4.0)


@register_test_case(module_factory=lambda: UpSampleNearest2d())
def UpSampleNearest2d_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 6, 12).to(torch.float64))

class UpSampleNearest2dSameSize(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, inputVec):
        return torch._C._nn.upsample_nearest2d(inputVec,
                                               output_size=[11, 11],
                                               scales_h=None,
                                               scales_w=None)


@register_test_case(module_factory=lambda: UpSampleNearest2dSameSize())
def UpSampleNearest2dStaticSize_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 4, 4))


class UpSampleNearest2dDiffSize(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1, -1, -1, -1], torch.float32, True)])
    def forward(self, inputVec):
        return torch._C._nn.upsample_nearest2d(inputVec,
                                               output_size=[8, 11],
                                               scales_h=None,
                                               scales_w=None)


@register_test_case(module_factory=lambda: UpSampleNearest2dDiffSize())
def UpSampleNearest2dDynamicSize_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 2, 2))


class UpSampleNearest2dDiffFactor(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1, -1, -1, -1], torch.float32, True)])
    def forward(self, inputVec):
        return torch._C._nn.upsample_nearest2d(inputVec,
                                               output_size=[6, 10],
                                               scales_h=2.3,
                                               scales_w=4.7)


@register_test_case(module_factory=lambda: UpSampleNearest2dDiffFactor())
def UpSampleNearest2dDynamicFactor_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 2, 2))


class UpSampleNearest2dSameFactor(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, inputVec):
        return torch._C._nn.upsample_nearest2d(inputVec,
                                               output_size=[8, 8],
                                               scales_h=2.0,
                                               scales_w=2.0)


@register_test_case(module_factory=lambda: UpSampleNearest2dSameFactor())
def UpSampleNearest2dStaticFactor_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4, 4))
class Conv1dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.float32, True),
        ([-1], torch.float32, True),
    ])
    def forward(self, inputVec, weight, bias):
        return torch.ops.aten.conv1d(inputVec,
                                     weight,
                                     bias=bias,
                                     stride=[1],
                                     padding=[0],
                                     dilation=[1],
                                     groups=1)
@register_test_case(module_factory=lambda: Conv1dModule())
def Conv1dModule_basic(module, tu: TestUtils):
    inputVec = tu.rand(2, 2, 6)
    weight = torch.randn(8, 2, 3)
    bias = torch.randn(8)
    module.forward(inputVec, weight, bias)

class Conv2dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
        ([-1, -1, -1, -1], torch.float32, True),
        ([-1], torch.float32, True),
    ])
    def forward(self, inputVec, weight, bias):
        return torch.ops.aten.conv2d(inputVec,
                                     weight,
                                     bias=bias,
                                     stride=[1, 1],
                                     padding=[0, 0],
                                     dilation=[1, 1],
                                     groups=1)
@register_test_case(module_factory=lambda: Conv2dModule())
def Conv2dModule_basic(module, tu: TestUtils):
    inputVec = tu.rand(2, 2, 6, 6)
    weight = torch.randn(8, 2, 3, 3)
    bias = torch.randn(8)
    module.forward(inputVec, weight, bias)

class Conv3dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1, -1], torch.float32, True),
        ([-1, -1, -1, -1, -1], torch.float32, True),
        ([-1], torch.float32, True),
    ])
    def forward(self, inputVec, weight, bias):
        return torch.ops.aten.conv3d(inputVec,
                                     weight,
                                     bias=bias,
                                     stride=[1, 1, 1],
                                     padding=[0, 0, 0],
                                     dilation=[1, 1, 1],
                                     groups=1)
@register_test_case(module_factory=lambda: Conv3dModule())
def Conv3dModule_basic(module, tu: TestUtils):
    inputVec = tu.rand(2, 2, 6, 6, 6)
    weight = torch.randn(8, 2, 3, 3, 3)
    bias = torch.randn(8)
    module.forward(inputVec, weight, bias)

class ConvTbcModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    # shapes from https://github.com/pytorch/pytorch/blob/3e8c8ce37bbfaafa8581fb48506c0a70ea54463d/test/nn/test_convolution.py#L623
    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.float32, True),
        ([-1], torch.float32, True),
    ])
    def forward(self, x, weight, bias):
        return torch.conv_tbc(x, weight, bias)

@register_test_case(module_factory=lambda: ConvTbcModule())
def ConvTbcModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(9, 4, 5), tu.rand(3, 5, 6), tu.rand(6))

class Conv2dQInt8Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.int8, True),
        ([-1, -1, -1, -1], torch.int8, True),
        ([-1], torch.float, True),
    ])
    def forward(self, inputVec, weight, bias):
        inputVec = torch._make_per_tensor_quantized_tensor(inputVec, 0.01, 7)
        inputVec = torch.dequantize(inputVec)

        weight = torch._make_per_tensor_quantized_tensor(weight, 0.01, 3)
        weight = torch.dequantize(weight)

        bias = torch.quantize_per_tensor(bias, 0.0001, 0, torch.qint32)
        bias = torch.dequantize(bias)

        return torch.ops.aten.conv2d(inputVec,
                                     weight,
                                     bias=bias,
                                     stride=[1, 1],
                                     padding=[0, 0],
                                     dilation=[1, 1],
                                     groups=1)
@register_test_case(module_factory=lambda: Conv2dQInt8Module())
def Conv2dQInt8Module_basic(module, tu: TestUtils):
    inputVec = tu.randint(2, 4, 7, 8, low=-128, high=127).to(torch.int8)
    weight = tu.randint(3, 4, 3, 2, low=-128, high=127).to(torch.int8)
    bias = torch.rand(3)
    module.forward(inputVec, weight, bias)
