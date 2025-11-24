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
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
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
        self.conv = torch.nn.Conv2d(
            in_channels=2,
            out_channels=10,
            kernel_size=3,
            padding=3,
            stride=2,
            dilation=3,
            bias=False,
        )
        self.train(False)

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.conv(x)


@register_test_case(module_factory=lambda: Conv2dWithPaddingDilationStrideModule())
def Conv2dWithPaddingDilationStrideModule_basic(module, tu: TestUtils):
    t = tu.rand(5, 2, 10, 20)
    module.forward(t)


class Conv2dWithPaddingDilationStrideStaticModule(torch.nn.Module):
    def __init__(self, out_channels, groups):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(
            in_channels=4,
            out_channels=out_channels,
            kernel_size=3,
            padding=3,
            stride=2,
            dilation=3,
            bias=False,
            groups=groups,
        )
        self.train(False)

    @export
    @annotate_args(
        [
            None,
            ([5, 4, 10, 20], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.conv(x)


@register_test_case(
    module_factory=lambda: Conv2dWithPaddingDilationStrideStaticModule(
        out_channels=10, groups=1
    )
)
def Conv2dWithPaddingDilationStrideStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 4, 10, 20))


@register_test_case(
    module_factory=lambda: Conv2dWithPaddingDilationStrideStaticModule(
        out_channels=4, groups=4
    )
)
def Conv2dWithPaddingDilationStrideStaticModule_depthwise(module, tu: TestUtils):
    module.forward(tu.rand(5, 4, 10, 20))


@register_test_case(
    module_factory=lambda: Conv2dWithPaddingDilationStrideStaticModule(
        out_channels=8, groups=4
    )
)
def Conv2dWithPaddingDilationStrideStaticModule_depthwise_multiplier(
    module, tu: TestUtils
):
    module.forward(tu.rand(5, 4, 10, 20))


@register_test_case(
    module_factory=lambda: Conv2dWithPaddingDilationStrideStaticModule(
        out_channels=4, groups=2
    )
)
def Conv2dWithPaddingDilationStrideStaticModule_grouped(module, tu: TestUtils):
    module.forward(tu.rand(5, 4, 10, 20))


@register_test_case(
    module_factory=lambda: Conv2dWithPaddingDilationStrideStaticModule(
        out_channels=8, groups=2
    )
)
def Conv2dWithPaddingDilationStrideStaticModule_grouped_multiplier(
    module, tu: TestUtils
):
    module.forward(tu.rand(5, 4, 10, 20))


class Conv2dWithSamePaddingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(2, 10, 3, bias=False, padding="same")
        self.train(False)

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.conv(x)


@register_test_case(module_factory=lambda: Conv2dWithSamePaddingModule())
def Conv2dWithSamePaddingModule_basic(module, tu: TestUtils):
    t = tu.rand(5, 2, 10, 20)
    module.forward(t)


class Conv2dWithValidPaddingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(2, 10, 3, bias=False, padding="valid")
        self.train(False)

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.conv(x)


@register_test_case(module_factory=lambda: Conv2dWithValidPaddingModule())
def Conv2dWithValidPaddingModule_basic(module, tu: TestUtils):
    t = tu.rand(5, 2, 10, 20)
    module.forward(t)


# ==============================================================================


class Convolution2DModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, inputVec, weight):
        return torch.ops.aten.convolution(
            inputVec,
            weight,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            transposed=False,
            output_padding=[0, 0],
            groups=1,
        )


@register_test_case(module_factory=lambda: Convolution2DModule())
def Convolution2DModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 3, 10, 10), tu.rand(3, 3, 2, 2))


class Convolution2DStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([3, 3, 10, 10], torch.float32, True),
            ([3, 3, 2, 2], torch.float32, True),
        ]
    )
    def forward(self, inputVec, weight):
        return torch.ops.aten.convolution(
            inputVec,
            weight,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            transposed=False,
            output_padding=[0, 0],
            groups=1,
        )


@register_test_case(module_factory=lambda: Convolution2DStaticModule())
def Convolution2DStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 3, 10, 10), tu.rand(3, 3, 2, 2))


class Convolution2DNextStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1, 80, 72, 72], torch.float32, True),
            ([80, 1, 7, 7], torch.float32, True),
            ([80], torch.float32, True),
        ]
    )
    def forward(self, inputVec, weight, bias):
        return torch.ops.aten.convolution(
            inputVec,
            weight,
            bias=bias,
            stride=[1, 1],
            padding=[3, 3],
            dilation=[1, 1],
            transposed=False,
            output_padding=[0, 0],
            groups=80,
        )


@register_test_case(module_factory=lambda: Convolution2DNextStaticModule())
def Convolution2DNextStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 80, 72, 72), tu.rand(80, 1, 7, 7), tu.rand(80))


class Convolution2DStridedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, inputVec, weight):
        return torch.ops.aten.convolution(
            inputVec,
            weight,
            bias=None,
            stride=[3, 3],
            padding=[2, 2],
            dilation=[1, 1],
            transposed=False,
            output_padding=[0, 0],
            groups=1,
        )


@register_test_case(module_factory=lambda: Convolution2DStridedModule())
def Convolution2DStridedModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 3, 10, 10), tu.rand(3, 3, 2, 2))


class _Convolution2DAllFalseModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, inputVec, weight):
        return torch.ops.aten._convolution(
            inputVec,
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
            allow_tf32=False,
        )


@register_test_case(module_factory=lambda: _Convolution2DAllFalseModule())
def _Convolution2DAllFalseModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 3, 10, 10), tu.rand(3, 3, 2, 2))


class _Convolution2DBenchmarkModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, inputVec, weight):
        return torch.ops.aten._convolution(
            inputVec,
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
            allow_tf32=False,
        )


@register_test_case(module_factory=lambda: _Convolution2DBenchmarkModule())
def _Convolution2DBenchmarkModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 3, 10, 10), tu.rand(3, 3, 2, 2))


class _Convolution2DDeterministicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, inputVec, weight):
        return torch.ops.aten._convolution(
            inputVec,
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
            allow_tf32=False,
        )


@register_test_case(module_factory=lambda: _Convolution2DDeterministicModule())
def _Convolution2DDeterministicModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 3, 10, 10), tu.rand(3, 3, 2, 2))


class _Convolution2DCudnnModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, inputVec, weight):
        return torch.ops.aten._convolution(
            inputVec,
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
            allow_tf32=False,
        )


@register_test_case(module_factory=lambda: _Convolution2DCudnnModule())
def _Convolution2DCudnnModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 3, 10, 10), tu.rand(3, 3, 2, 2))


class _Convolution2DTF32Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, inputVec, weight):
        return torch.ops.aten._convolution(
            inputVec,
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
            allow_tf32=True,
        )


@register_test_case(module_factory=lambda: _Convolution2DTF32Module())
def _Convolution2DTF32Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 3, 10, 10), tu.rand(3, 3, 2, 2))


class _ConvolutionDeprecated2DAllFalseModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, inputVec, weight):
        return torch.ops.aten._convolution(
            inputVec,
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
        )


@register_test_case(module_factory=lambda: _ConvolutionDeprecated2DAllFalseModule())
def _ConvolutionDeprecated2DAllFalseModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 3, 10, 10), tu.rand(3, 3, 2, 2))


class _ConvolutionDeprecated2DBenchmarkModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, inputVec, weight):
        return torch.ops.aten._convolution(
            inputVec,
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
        )


@register_test_case(module_factory=lambda: _ConvolutionDeprecated2DBenchmarkModule())
def _ConvolutionDeprecated2DBenchmarkModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 3, 10, 10), tu.rand(3, 3, 2, 2))


class _ConvolutionDeprecated2DDeterministicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, inputVec, weight):
        return torch.ops.aten._convolution(
            inputVec,
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
        )


@register_test_case(
    module_factory=lambda: _ConvolutionDeprecated2DDeterministicModule()
)
def _ConvolutionDeprecated2DDeterministicModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 3, 10, 10), tu.rand(3, 3, 2, 2))


class _ConvolutionDeprecated2DCudnnModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, inputVec, weight):
        return torch.ops.aten._convolution(
            inputVec,
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
        )


@register_test_case(module_factory=lambda: _ConvolutionDeprecated2DCudnnModule())
def _ConvolutionDeprecated2DCudnnModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 3, 10, 10), tu.rand(3, 3, 2, 2))


class ConvolutionModule2DGroups(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, inputVec, weight):
        return torch.ops.aten.convolution(
            inputVec,
            weight,
            bias=None,
            stride=[3, 3],
            padding=[2, 2],
            dilation=[1, 1],
            transposed=False,
            output_padding=[0, 0],
            groups=4,
        )


@register_test_case(module_factory=lambda: ConvolutionModule2DGroups())
def ConvolutionModule2DGroups_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 32, 4, 4), tu.rand(32, 8, 3, 3))


class ConvolutionModule3DGroups(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1, -1], torch.float32, True),
            ([-1, -1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, inputVec, weight):
        return torch.ops.aten.convolution(
            inputVec,
            weight,
            bias=None,
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            dilation=[1, 1, 1],
            transposed=False,
            output_padding=[0, 0, 0],
            groups=2,
        )


@register_test_case(module_factory=lambda: ConvolutionModule3DGroups())
def ConvolutionModule3DGroups_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4, 6, 6, 6), tu.rand(8, 2, 3, 3, 3))


class ConvolutionModule3DGroupsStrided(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1, -1], torch.float32, True),
            ([-1, -1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, inputVec, weight):
        return torch.ops.aten.convolution(
            inputVec,
            weight,
            bias=None,
            stride=[2, 2, 2],
            padding=[1, 1, 1],
            dilation=[1, 1, 1],
            transposed=False,
            output_padding=[0, 0, 0],
            groups=4,
        )


@register_test_case(module_factory=lambda: ConvolutionModule3DGroupsStrided())
def ConvolutionModule3DGroupsStrided_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 8, 8, 8, 8), tu.rand(16, 2, 3, 3, 3))


class ConvolutionModule3DGroupsDilated(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1, -1], torch.float32, True),
            ([-1, -1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, inputVec, weight):
        return torch.ops.aten.convolution(
            inputVec,
            weight,
            bias=None,
            stride=[1, 1, 1],
            padding=[2, 2, 2],
            dilation=[2, 2, 2],
            transposed=False,
            output_padding=[0, 0, 0],
            groups=2,
        )


@register_test_case(module_factory=lambda: ConvolutionModule3DGroupsDilated())
def ConvolutionModule3DGroupsDilated_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4, 8, 8, 8), tu.rand(8, 2, 3, 3, 3))


# ==============================================================================


class ConvolutionModule2DTranspose(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, inputVec, weight):
        return torch.ops.aten.convolution(
            inputVec,
            weight,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            dilation=[1, 1],
            transposed=True,
            output_padding=[0, 0],
            groups=1,
        )


@register_test_case(module_factory=lambda: ConvolutionModule2DTranspose())
def ConvolutionModule2DTranspose_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 3, 4, 4), tu.rand(3, 3, 2, 2))


class ConvolutionModule2DTransposeStrided(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, inputVec, weight):
        return torch.ops.aten.convolution(
            inputVec,
            weight,
            bias=None,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            transposed=True,
            output_padding=[0, 0],
            groups=1,
        )


@register_test_case(module_factory=lambda: ConvolutionModule2DTransposeStrided())
def ConvolutionModule2DTransposeStrided_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 2, 5, 6), tu.rand(2, 5, 2, 2))


class ConvolutionModule2DTransposeStridedStatic(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([5, 2, 5, 6], torch.float32, True),
            ([2, 5, 2, 2], torch.float32, True),
        ]
    )
    def forward(self, inputVec, weight):
        return torch.ops.aten.convolution(
            inputVec,
            weight,
            bias=None,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            transposed=True,
            output_padding=[0, 0],
            groups=1,
        )


@register_test_case(module_factory=lambda: ConvolutionModule2DTransposeStridedStatic())
def ConvolutionModule2DTransposeStridedStatic_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 2, 5, 6), tu.rand(2, 5, 2, 2))


class ConvolutionModule2DTransposeNonUnitOutputPadding(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, inputVec, weight):
        return torch.ops.aten.convolution(
            inputVec,
            weight,
            bias=None,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            transposed=True,
            output_padding=[1, 1],
            groups=1,
        )


@register_test_case(
    module_factory=lambda: ConvolutionModule2DTransposeNonUnitOutputPadding()
)
def ConvolutionModule2DTransposeNonUnitOutputPadding_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 2, 4, 4), tu.rand(2, 2, 3, 3))


class Conv_Transpose1dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, inputVec, weight):
        return torch.ops.aten.conv_transpose1d(
            inputVec,
            weight,
            bias=None,
            stride=[2],
            padding=[1],
            dilation=[1],
            output_padding=[0],
            groups=1,
        )


@register_test_case(module_factory=lambda: Conv_Transpose1dModule())
def Conv_Transpose1dModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 2, 6), tu.rand(2, 5, 2))


class Conv_Transpose1dStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([5, 2, 6], torch.float32, True),
            ([2, 5, 2], torch.float32, True),
        ]
    )
    def forward(self, inputVec, weight):
        return torch.ops.aten.conv_transpose1d(
            inputVec,
            weight,
            bias=None,
            stride=[2],
            padding=[1],
            dilation=[1],
            output_padding=[0],
            groups=1,
        )


@register_test_case(module_factory=lambda: Conv_Transpose1dStaticModule())
def Conv_Transpose1dStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 2, 6), tu.rand(2, 5, 2))


class Conv_Transpose2dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, inputVec, weight):
        return torch.ops.aten.conv_transpose2d(
            inputVec,
            weight,
            bias=None,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            output_padding=[0, 0],
            groups=1,
        )


@register_test_case(module_factory=lambda: Conv_Transpose2dModule())
def Conv_Transpose2dModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 2, 5, 6), tu.rand(2, 5, 2, 2))


class Conv_Transpose2dStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([5, 2, 5, 6], torch.float32, True),
            ([2, 5, 2, 2], torch.float32, True),
        ]
    )
    def forward(self, inputVec, weight):
        return torch.ops.aten.conv_transpose2d(
            inputVec,
            weight,
            bias=None,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            output_padding=[0, 0],
            groups=1,
        )


@register_test_case(module_factory=lambda: Conv_Transpose2dStaticModule())
def Conv_Transpose2dStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 2, 5, 6), tu.rand(2, 5, 2, 2))


class Conv_Transpose3dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1, -1], torch.float32, True),
            ([-1, -1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, inputVec, weight):
        return torch.ops.aten.conv_transpose3d(
            inputVec,
            weight,
            bias=None,
            stride=[2, 2, 2],
            padding=[1, 1, 1],
            dilation=[1, 1, 1],
            output_padding=[0, 0, 0],
            groups=1,
        )


@register_test_case(module_factory=lambda: Conv_Transpose3dModule())
def Conv_Transpose3dModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 2, 5, 6, 7), tu.rand(2, 5, 2, 2, 2))


class Conv_Transpose3dStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([5, 2, 5, 6, 7], torch.float32, True),
            ([2, 5, 2, 2, 2], torch.float32, True),
        ]
    )
    def forward(self, inputVec, weight):
        return torch.ops.aten.conv_transpose3d(
            inputVec,
            weight,
            bias=None,
            stride=[2, 2, 2],
            padding=[1, 1, 1],
            dilation=[1, 1, 1],
            output_padding=[0, 0, 0],
            groups=1,
        )


@register_test_case(module_factory=lambda: Conv_Transpose3dStaticModule())
def Conv_Transpose3dStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 2, 5, 6, 7), tu.rand(2, 5, 2, 2, 2))


class UpSampleNearest2d(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float64, True),
        ]
    )
    def forward(self, input):
        return torch.ops.aten.upsample_nearest2d(
            input, output_size=[18, 48], scales_h=3.0, scales_w=4.0
        )


@register_test_case(module_factory=lambda: UpSampleNearest2d())
def UpSampleNearest2d_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 6, 12).to(torch.float64))


class UpSampleNearest2dSameSize(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, inputVec):
        return torch._C._nn.upsample_nearest2d(
            inputVec, output_size=[11, 11], scales_h=None, scales_w=None
        )


@register_test_case(module_factory=lambda: UpSampleNearest2dSameSize())
def UpSampleNearest2dStaticSize_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 4, 4))


class UpSampleNearest2dDiffSize(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1, -1, -1, -1], torch.float32, True)])
    def forward(self, inputVec):
        return torch._C._nn.upsample_nearest2d(
            inputVec, output_size=[8, 11], scales_h=None, scales_w=None
        )


@register_test_case(module_factory=lambda: UpSampleNearest2dDiffSize())
def UpSampleNearest2dDynamicSize_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 2, 2))


class UpSampleNearest2dDiffFactor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1, -1, -1, -1], torch.float32, True)])
    def forward(self, inputVec):
        return torch._C._nn.upsample_nearest2d(
            inputVec, output_size=[6, 10], scales_h=2.3, scales_w=4.7
        )


@register_test_case(module_factory=lambda: UpSampleNearest2dDiffFactor())
def UpSampleNearest2dDynamicFactor_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 2, 2))


class UpSampleNearest2dSameFactor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, inputVec):
        return torch._C._nn.upsample_nearest2d(
            inputVec, output_size=[8, 8], scales_h=2.0, scales_w=2.0
        )


@register_test_case(module_factory=lambda: UpSampleNearest2dSameFactor())
def UpSampleNearest2dStaticFactor_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4, 4))


class UpSampleNearest2dVecNoneShape(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float64, True),
        ]
    )
    def forward(self, input):
        return torch.ops.aten.upsample_nearest2d.vec(
            input, output_size=None, scale_factors=[3.66, 4.2]
        )


@register_test_case(module_factory=lambda: UpSampleNearest2dVecNoneShape())
def UpSampleNearest2dVecNoneShape_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 6, 12).to(torch.float64))


class UpSampleNearest2dVecNoneScales(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float64, True),
        ]
    )
    def forward(self, input):
        return torch.ops.aten.upsample_nearest2d.vec(
            input,
            output_size=[18, 48],
            scale_factors=None,
        )


@register_test_case(module_factory=lambda: UpSampleNearest2dVecNoneScales())
def UpSampleNearest2dVecNoneScales_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 6, 12).to(torch.float64))


class UpSampleNearest1dVecNoneShape(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float64, True),
        ]
    )
    def forward(self, input):
        return torch.ops.aten.upsample_nearest1d.vec(
            input, output_size=None, scale_factors=[3.0]
        )


@register_test_case(module_factory=lambda: UpSampleNearest1dVecNoneShape())
def UpSampleNearest1dVecNoneShape_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 6).to(torch.float64))


class UpSampleNearest1dVecNoneScales(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float64, True),
        ]
    )
    def forward(self, input):
        return torch.ops.aten.upsample_nearest1d.vec(input, [18], None)


@register_test_case(module_factory=lambda: UpSampleNearest1dVecNoneScales())
def UpSampleNearest1dVecNoneScales_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 6).to(torch.float64))


class Conv1dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1, -1], torch.float32, True),
            ([-1], torch.float32, True),
        ]
    )
    def forward(self, inputVec, weight, bias):
        return torch.ops.aten.conv1d(
            inputVec, weight, bias=bias, stride=[1], padding=[0], dilation=[1], groups=1
        )


@register_test_case(module_factory=lambda: Conv1dModule())
def Conv1dModule_basic(module, tu: TestUtils):
    inputVec = tu.rand(2, 2, 6)
    weight = torch.randn(8, 2, 3)
    bias = torch.randn(8)
    module.forward(inputVec, weight, bias)


class Conv1dDepthwiseWithPaddingDilationStrideStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 4, 6], torch.float32, True),
            ([4, 1, 3], torch.float32, True),
        ]
    )
    def forward(self, inputVec, weight):
        return torch.ops.aten.conv1d(
            inputVec, weight, bias=None, stride=[1], padding=[4], dilation=[1], groups=4
        )


@register_test_case(
    module_factory=lambda: Conv1dDepthwiseWithPaddingDilationStrideStaticModule()
)
def Conv1dDepthwiseWithPaddingDilationStrideStaticModule_basic(module, tu: TestUtils):
    inputVec = tu.rand(2, 4, 6)
    weight = torch.randn(4, 1, 3)
    module.forward(inputVec, weight)


class Conv1dWithSamePaddingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv1d(2, 10, 3, bias=False, padding="same")
        self.train(False)

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.conv(x)


@register_test_case(module_factory=lambda: Conv1dWithSamePaddingModule())
def Conv1dWithSamePaddingModule_basic(module, tu: TestUtils):
    t = tu.rand(5, 2, 10)
    module.forward(t)


class Conv1dWithValidPaddingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1, -1], torch.float32, True),
            ([-1], torch.float32, True),
        ]
    )
    def forward(self, inputVec, weight, bias):
        return torch.ops.aten.conv1d(
            inputVec,
            weight,
            bias=bias,
            stride=[1],
            padding="valid",
            dilation=[1],
            groups=1,
        )


@register_test_case(module_factory=lambda: Conv1dWithValidPaddingModule())
def Conv1dWithValidPaddingModule_basic(module, tu: TestUtils):
    inputVec = tu.rand(2, 2, 6)
    weight = torch.randn(8, 2, 3)
    bias = torch.randn(8)
    module.forward(inputVec, weight, bias)


class Conv1dGroupModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1, -1], torch.float32, True),
            ([-1], torch.float32, True),
        ]
    )
    def forward(self, inputVec, weight, bias):
        return torch.ops.aten.conv1d(
            inputVec, weight, bias=bias, stride=[1], padding=[0], dilation=[1], groups=2
        )


@register_test_case(module_factory=lambda: Conv1dGroupModule())
def Conv1dGroupModule_basic(module, tu: TestUtils):
    inputVec = tu.rand(2, 4, 6)
    weight = torch.randn(8, 2, 3)
    bias = torch.randn(8)
    module.forward(inputVec, weight, bias)


class Conv2dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
            ([-1, -1, -1, -1], torch.float32, True),
            ([-1], torch.float32, True),
        ]
    )
    def forward(self, inputVec, weight, bias):
        return torch.ops.aten.conv2d(
            inputVec,
            weight,
            bias=bias,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
        )


@register_test_case(module_factory=lambda: Conv2dModule())
def Conv2dModule_basic(module, tu: TestUtils):
    inputVec = tu.rand(2, 2, 6, 6)
    weight = torch.randn(8, 2, 3, 3)
    bias = torch.randn(8)
    module.forward(inputVec, weight, bias)


class Conv2dFP16NoBiasModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float16, True),
            ([-1, -1, -1, -1], torch.float16, True),
        ]
    )
    def forward(self, inputVec, weight):
        return torch.ops.aten.conv2d(
            inputVec,
            weight,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=1,
        )


@register_test_case(module_factory=lambda: Conv2dFP16NoBiasModule())
def Conv2dFP16NoBiasModule_basic(module, tu: TestUtils):
    inputVec = tu.rand(2, 2, 6, 6).to(torch.float16)
    weight = torch.randn(8, 2, 3, 3).to(torch.float16)
    module.forward(inputVec, weight)


class Conv3dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1, -1], torch.float32, True),
            ([-1, -1, -1, -1, -1], torch.float32, True),
            ([-1], torch.float32, True),
        ]
    )
    def forward(self, inputVec, weight, bias):
        return torch.ops.aten.conv3d(
            inputVec,
            weight,
            bias=bias,
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            dilation=[1, 1, 1],
            groups=1,
        )


@register_test_case(module_factory=lambda: Conv3dModule())
def Conv3dModule_basic(module, tu: TestUtils):
    inputVec = tu.rand(2, 2, 6, 6, 6)
    weight = torch.randn(8, 2, 3, 3, 3)
    bias = torch.randn(8)
    module.forward(inputVec, weight, bias)


class Conv3dWithSamePaddingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1, -1], torch.float32, True),
            ([-1, -1, -1, -1, -1], torch.float32, True),
            ([-1], torch.float32, True),
        ]
    )
    def forward(self, inputVec, weight, bias):
        return torch.ops.aten.conv3d(
            inputVec,
            weight,
            bias=bias,
            stride=[1, 1, 1],
            padding="same",
            dilation=[1, 1, 1],
            groups=1,
        )


@register_test_case(module_factory=lambda: Conv3dWithSamePaddingModule())
def Conv3dWithSamePaddingModule_basic(module, tu: TestUtils):
    inputVec = tu.rand(2, 2, 6, 6, 6)
    weight = torch.randn(8, 2, 3, 3, 3)
    bias = torch.randn(8)
    module.forward(inputVec, weight, bias)


class Conv3dWithValidPaddingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1, -1], torch.float32, True),
            ([-1, -1, -1, -1, -1], torch.float32, True),
            ([-1], torch.float32, True),
        ]
    )
    def forward(self, inputVec, weight, bias):
        return torch.ops.aten.conv3d(
            inputVec,
            weight,
            bias=bias,
            stride=[1, 1, 1],
            padding="valid",
            dilation=[1, 1, 1],
            groups=1,
        )


@register_test_case(module_factory=lambda: Conv3dWithValidPaddingModule())
def Conv3dWithValidPaddingModule_basic(module, tu: TestUtils):
    inputVec = tu.rand(2, 2, 6, 6, 6)
    weight = torch.randn(8, 2, 3, 3, 3)
    bias = torch.randn(8)
    module.forward(inputVec, weight, bias)


class ConvTbcModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    # shapes from https://github.com/pytorch/pytorch/blob/3e8c8ce37bbfaafa8581fb48506c0a70ea54463d/test/nn/test_convolution.py#L623
    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1, -1], torch.float32, True),
            ([-1], torch.float32, True),
        ]
    )
    def forward(self, x, weight, bias):
        return torch.conv_tbc(x, weight, bias)


@register_test_case(module_factory=lambda: ConvTbcModule())
def ConvTbcModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(9, 4, 5), tu.rand(3, 5, 6), tu.rand(6))


# For DQ-Q fake quantization ops
import torch.ao.quantization.fx._decomposed


class Conv2dQInt8ModuleBase(torch.nn.Module):
    def __init__(self, groups=1):
        self.groups = groups
        super().__init__()

    def _forward(self, input, weight, bias):
        input = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            input, 0.01, 7, -128, 127, torch.int8
        )
        weight = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            weight, 0.01, 3, -128, 127, torch.int8
        )
        bias = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            bias, 1, 0, -1000, 1000, torch.int32
        )

        conv = torch.ops.aten.conv2d(
            input,
            weight,
            bias=bias,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=self.groups,
        )

        # Use int32 to avoid overflows
        return torch.ops.quantized_decomposed.quantize_per_tensor.default(
            conv, 1, 0, -(2**31), 2**31 - 1, torch.int32
        )


class Conv2dQInt8ModuleDyn(Conv2dQInt8ModuleBase):
    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.int8, True),
            ([-1, -1, -1, -1], torch.int8, True),
            ([-1], torch.int32, True),
        ]
    )
    def forward(self, inputVec, weight, bias):
        return self._forward(inputVec, weight, bias)


class Conv2dQInt8ModuleStatic(Conv2dQInt8ModuleBase):
    @export
    @annotate_args(
        [
            None,
            ([2, 3, 12, 12], torch.int8, True),
            ([3, 1, 5, 3], torch.int8, True),
            ([3], torch.int32, True),
        ]
    )
    def forward(self, inputVec, weight, bias):
        return self._forward(inputVec, weight, bias)


class Conv2dQInt8ModuleStatic_MoreOutChannels(Conv2dQInt8ModuleBase):
    @export
    @annotate_args(
        [
            None,
            ([2, 3, 12, 12], torch.int8, True),
            ([6, 1, 5, 3], torch.int8, True),
            ([6], torch.int32, True),
        ]
    )
    def forward(self, inputVec, weight, bias):
        return self._forward(inputVec, weight, bias)


@register_test_case(module_factory=lambda: Conv2dQInt8ModuleDyn())
def Conv2dQInt8Module_basic(module, tu: TestUtils):
    inputVec = tu.randint(2, 4, 7, 8, low=-128, high=127).to(torch.int8)
    weight = tu.randint(3, 4, 3, 2, low=-128, high=127).to(torch.int8)
    bias = tu.randint(3, low=-1000, high=1000).to(torch.int32)
    module.forward(inputVec, weight, bias)


@register_test_case(module_factory=lambda: Conv2dQInt8ModuleDyn(groups=2))
def Conv2dQInt8Module_grouped(module, tu: TestUtils):
    inputVec = tu.randint(2, 8, 7, 8, low=-128, high=127).to(torch.int8)
    weight = tu.randint(6, 4, 3, 2, low=-128, high=127).to(torch.int8)
    bias = tu.randint(6, low=-1000, high=1000).to(torch.int32)
    module.forward(inputVec, weight, bias)


@register_test_case(module_factory=lambda: Conv2dQInt8ModuleStatic(groups=3))
def Conv2dQInt8Module_depthwise(module, tu: TestUtils):
    inputVec = tu.randint(2, 3, 12, 12, low=-128, high=127).to(torch.int8)
    weight = tu.randint(3, 1, 5, 3, low=-128, high=127).to(torch.int8)
    bias = tu.randint(3, low=-1000, high=1000).to(torch.int32)
    module.forward(inputVec, weight, bias)


@register_test_case(
    module_factory=lambda: Conv2dQInt8ModuleStatic_MoreOutChannels(groups=3)
)
def Conv2dQInt8Module_not_depthwise(module, tu: TestUtils):
    inputVec = tu.randint(2, 3, 12, 12, low=-128, high=127).to(torch.int8)
    weight = tu.randint(6, 1, 5, 3, low=-128, high=127).to(torch.int8)
    bias = tu.randint(6, low=-1000, high=1000).to(torch.int32)
    module.forward(inputVec, weight, bias)


class ConvTranspose2DQInt8Module(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.int8, True),
            ([-1, -1, -1, -1], torch.int8, True),
            ([-1], torch.float, True),
        ]
    )
    def forward(self, input, weight, bias):
        input = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            input, 0.01, -25, -128, 127, torch.int8
        )
        weight = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            weight, 0.01, 50, -128, 127, torch.int8
        )

        res = torch.ops.aten.convolution(
            input,
            weight,
            bias=bias,
            stride=[2, 1],
            padding=[1, 1],
            dilation=[1, 1],
            transposed=True,
            output_padding=[0, 0],
            groups=1,
        )

        # Use int32 to avoid overflows
        return torch.ops.quantized_decomposed.quantize_per_tensor.default(
            res, 1, 0, -(2**31), 2**31 - 1, torch.int32
        )


@register_test_case(module_factory=lambda: ConvTranspose2DQInt8Module())
def ConvTranspose2DQInt8_basic(module, tu: TestUtils):
    N = 10
    Cin = 5
    Cout = 7
    Hin = 10
    Win = 8
    Hker = 3
    Wker = 2
    module.forward(
        tu.randint(N, Cin, Hin, Win, low=-128, high=127).to(torch.int8),
        tu.randint(Cin, Cout, Hker, Wker, low=-128, high=127).to(torch.int8),
        torch.rand(Cout),
    )


class Conv2dQInt8PerChannelModuleBase(torch.nn.Module):
    def __init__(self, groups=1):
        self.groups = groups
        super().__init__()

    def _forward(self, inputVec, weight, scales, zeropoints, bias):
        inputVec = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            inputVec, 0.01, 7, -128, 127, torch.int8
        )
        weight = torch.ops.quantized_decomposed.dequantize_per_channel.default(
            weight, scales, zeropoints, 0, -128, 127, torch.int8
        )

        conv = torch.ops.aten.conv2d(
            inputVec,
            weight,
            bias=bias,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            groups=self.groups,
        )

        # Use int32 to avoid overflows
        return torch.ops.quantized_decomposed.quantize_per_tensor.default(
            conv, 1, 0, -(2**31), 2**31 - 1, torch.int32
        )


class Conv2dQInt8PerChannelModuleDyn(Conv2dQInt8PerChannelModuleBase):
    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.int8, True),
            ([-1, -1, -1, -1], torch.int8, True),
            ([-1], torch.float, True),
            ([-1], torch.int8, True),
            ([-1], torch.float, True),
        ]
    )
    def forward(self, inputVec, weight, scales, zeropoints, bias):
        return self._forward(inputVec, weight, scales, zeropoints, bias)


class Conv2dQInt8PerChannelModuleStatic(Conv2dQInt8PerChannelModuleBase):
    @export
    @annotate_args(
        [
            None,
            ([2, 3, 12, 12], torch.int8, True),
            ([3, 1, 5, 3], torch.int8, True),
            ([3], torch.float, True),
            ([3], torch.int8, True),
            ([3], torch.float, True),
        ]
    )
    def forward(self, inputVec, weight, scales, zeropoints, bias):
        return self._forward(inputVec, weight, scales, zeropoints, bias)


@register_test_case(module_factory=lambda: Conv2dQInt8PerChannelModuleDyn())
def Conv2dQInt8PerChannelModule_basic(module, tu: TestUtils):
    inputVec = tu.randint(2, 4, 7, 8, low=-128, high=127).to(torch.int8)
    weight = tu.randint(3, 4, 3, 2, low=-128, high=127).to(torch.int8)
    scales = tu.rand(3)
    zeropoints = tu.rand(3).to(torch.int8)
    bias = torch.rand(3)
    module.forward(inputVec, weight, scales, zeropoints, bias)


@register_test_case(module_factory=lambda: Conv2dQInt8PerChannelModuleDyn(groups=2))
def Conv2dQInt8PerChannelModule_grouped(module, tu: TestUtils):
    inputVec = tu.randint(2, 8, 7, 8, low=-128, high=127).to(torch.int8)
    weight = tu.randint(6, 4, 3, 2, low=-128, high=127).to(torch.int8)
    scales = tu.rand(6)
    zeropoints = tu.rand(6).to(torch.int8)
    bias = torch.rand(6)
    module.forward(inputVec, weight, scales, zeropoints, bias)


@register_test_case(module_factory=lambda: Conv2dQInt8PerChannelModuleStatic(groups=3))
def Conv2dQInt8PerChannelModule_depthwise(module, tu: TestUtils):
    inputVec = tu.randint(2, 3, 12, 12, low=-128, high=127).to(torch.int8)
    weight = tu.randint(3, 1, 5, 3, low=-128, high=127).to(torch.int8)
    scales = tu.rand(3)
    zeropoints = tu.rand(3).to(torch.int8)
    bias = torch.rand(3)
    module.forward(inputVec, weight, scales, zeropoints, bias)


# torchvision.deform_conv2d

import torchvision

# This section defines a torch->onnx path for this torchvision op so we can test the onnx paths e2e.

# Create symbolic function
from torch.onnx.symbolic_helper import parse_args, _get_tensor_sizes


@parse_args("v", "v", "v", "v", "v", "i", "i", "i", "i", "i", "i", "i", "i", "b")
def symbolic_deform_conv2d_forward(
    g,
    input,
    weight,
    offset,
    mask,
    bias,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    groups,
    offset_groups,
    use_mask,
):
    args = [input, weight, offset, bias]
    if use_mask:
        args.append(mask)
    weight_size = _get_tensor_sizes(weight)
    kwargs = {
        "dilations_i": [dilation_h, dilation_w],
        "group_i": groups,
        "kernel_shape_i": weight_size[2:],
        "offset_group_i": offset_groups,
        # NB: ONNX supports asymmetric padding, whereas PyTorch supports only
        # symmetric padding
        "pads_i": [pad_h, pad_w, pad_h, pad_w],
        "strides_i": [stride_h, stride_w],
    }
    return g.op("DeformConv", *args, **kwargs)


# Register symbolic function
from torch.onnx import register_custom_op_symbolic

register_custom_op_symbolic(
    "torchvision::deform_conv2d", symbolic_deform_conv2d_forward, 19
)

N = 1
Cin = 1
Hin = 7
Win = 6
Cout = 1
Hker = 2
Wker = 2
offset_groups = 1
Hout = 6
Wout = 5
offset_dim1 = 2 * offset_groups * Hker * Wker


class DeformableConvModule(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([N, Cin, Hin, Win], torch.float32, True),
            ([N, offset_dim1, Hout, Wout], torch.float32, True),
            ([Cout, Cin, Hker, Wker], torch.float32, True),
        ]
    )
    def forward(self, input, offset, weight):
        return torchvision.ops.deform_conv2d(input, offset, weight)


@register_test_case(module_factory=lambda: DeformableConvModule())
def DeformConv2D_basic(module, tu: TestUtils):
    input = tu.rand(N, Cin, Hin, Win)
    offset = tu.rand(N, offset_dim1, Hout, Wout)
    weight = tu.rand(Cout, Cin, Hker, Wker)
    module.forward(input, offset, weight)


class ConvolutionModule2DGroupedTranspose(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1, 2, 5, 7], torch.float32, True),
            ([2, 2, 3, 3], torch.float32, True),
            ([4], torch.float32, True),
        ]
    )
    def forward(self, inputVec, weight, bias):
        return torch.ops.aten.convolution(
            inputVec,
            weight,
            bias=bias,
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            transposed=True,
            output_padding=[0, 0],
            groups=2,
        )


@register_test_case(module_factory=lambda: ConvolutionModule2DGroupedTranspose())
def ConvolutionModule2DGroupedTranspose_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 2, 5, 7), tu.rand(2, 2, 3, 3), tu.rand(4))


class TransposedConv1dNegativePadding(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1, 1, 7], torch.float32, True),
            ([1, 2, 3], torch.float32, True),
            ([2], torch.float32, True),
        ]
    )
    def forward(self, inputVec, weight, bias):
        return torch.ops.aten.convolution(
            inputVec,
            weight,
            bias=bias,
            stride=[4],
            padding=[3],
            dilation=[1],
            transposed=True,
            output_padding=[0],
            groups=1,
        )


@register_test_case(module_factory=lambda: TransposedConv1dNegativePadding())
def TransposedConv1dNegativePadding_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 7), tu.rand(1, 2, 3), tu.rand(2))


class TransposedConv1dNegativePaddingUnitStrideDyn(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
            ([1, 2, 3], torch.float32, True),
            ([2], torch.float32, True),
        ]
    )
    def forward(self, inputVec, weight, bias):
        return torch.ops.aten.convolution(
            inputVec,
            weight,
            bias=bias,
            stride=[1],
            padding=[3],
            dilation=[1],
            transposed=True,
            output_padding=[0],
            groups=1,
        )


@register_test_case(
    module_factory=lambda: TransposedConv1dNegativePaddingUnitStrideDyn()
)
def TransposedConv1dNegativePaddingUnitStrideDyn_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 7), tu.rand(1, 2, 3), tu.rand(2))


class TransposedConv1dNegativePaddingLarge(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1, 17, 5], torch.float32, True),
            ([17, 6, 3], torch.float32, True),
            ([6], torch.float32, True),
        ]
    )
    def forward(self, inputVec, weight, bias):
        return torch.ops.aten.convolution(
            inputVec,
            weight,
            bias=bias,
            stride=[7],
            padding=[10],
            dilation=[4],
            transposed=True,
            output_padding=[0],
            groups=1,
        )


@register_test_case(module_factory=lambda: TransposedConv1dNegativePaddingLarge())
def TransposedConv1dNegativePaddingLarge_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 17, 5), tu.rand(17, 6, 3), tu.rand(6))


class TransposedConv2dNegativePadding(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1, 1, 4, 7], torch.float32, True),
            ([1, 2, 3, 3], torch.float32, True),
            ([2], torch.float32, True),
        ]
    )
    def forward(self, inputVec, weight, bias):
        return torch.ops.aten.convolution(
            inputVec,
            weight,
            bias=bias,
            stride=[1, 1],
            padding=[0, 3],
            dilation=[1, 1],
            transposed=True,
            output_padding=[0, 0],
            groups=1,
        )


@register_test_case(module_factory=lambda: TransposedConv2dNegativePadding())
def TransposedConv2dNegativePadding_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 4, 7), tu.rand(1, 2, 3, 3), tu.rand(2))


class TransposedConv2dPositiveAndNegativePadding(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1, 1, 4, 7], torch.float32, True),
            ([1, 2, 3, 3], torch.float32, True),
            ([2], torch.float32, True),
        ]
    )
    def forward(self, inputVec, weight, bias):
        return torch.ops.aten.convolution(
            inputVec,
            weight,
            bias=bias,
            stride=[4, 4],
            padding=[0, 3],
            dilation=[1, 1],
            transposed=True,
            output_padding=[0, 0],
            groups=1,
        )


@register_test_case(module_factory=lambda: TransposedConv2dPositiveAndNegativePadding())
def TransposedConv2dPositiveAndNegativePadding_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 4, 7), tu.rand(1, 2, 3, 3), tu.rand(2))


class TransposedConv3dNegativePadding(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([4, 1, 8, 13, 17], torch.float32, True),
            ([1, 1, 3, 7, 3], torch.float32, True),
            ([1], torch.float32, True),
        ]
    )
    def forward(self, inputVec, weight, bias):
        return torch.ops.aten.convolution(
            inputVec,
            weight,
            bias=bias,
            stride=[1, 5, 3],
            padding=[2, 1, 3],
            dilation=[1, 2, 1],
            transposed=True,
            output_padding=[0, 0, 0],
            groups=1,
        )


@register_test_case(module_factory=lambda: TransposedConv3dNegativePadding())
def TransposedConv3dNegativePadding_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 1, 8, 13, 17), tu.rand(1, 1, 3, 7, 3), tu.rand(1))
