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
