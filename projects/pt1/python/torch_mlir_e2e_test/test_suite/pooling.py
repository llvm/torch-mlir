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
    @annotate_args(
        [
            None,
            ([1, 512, 7, 7], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.aap2d(x)


@register_test_case(
    module_factory=lambda: AdaptiveAvgPool2dNonUnitOutputSizeStaticModule()
)
def AdaptiveAvgPool2dNonUnitOutputSizeStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 512, 7, 7))


class AdaptiveAvgPool2dNonUnitOutputSizeDynamicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.aap2d = torch.nn.AdaptiveAvgPool2d((7, 7))

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.aap2d(x)


@register_test_case(
    module_factory=lambda: AdaptiveAvgPool2dNonUnitOutputSizeDynamicModule()
)
def AdaptiveAvgPool2dNonUnitOutputSizeDynamicModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 512, 7, 7))


class AdaptiveAvgPool2dOutputSizeDivisibleByInputDynamicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.aap2d = torch.nn.AdaptiveAvgPool2d((5, 7))

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.aap2d(x)


@register_test_case(
    module_factory=lambda: AdaptiveAvgPool2dOutputSizeDivisibleByInputDynamicModule()
)
def AdaptiveAvgPool2dOutputSizeDivisibleByInputDynamicModule_basic(
    module, tu: TestUtils
):
    module.forward(tu.rand(1, 512, 15, 28))


class AdaptiveAvgPool2dOutputSizeDivisibleByInputStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.aap2d = torch.nn.AdaptiveAvgPool2d((3, 7))

    @export
    @annotate_args(
        [
            None,
            ([1, 512, 15, 14], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.aap2d(x)


@register_test_case(
    module_factory=lambda: AdaptiveAvgPool2dOutputSizeDivisibleByInputStaticModule()
)
def AdaptiveAvgPool2dOutputSizeDivisibleByInputStaticModule_basic(
    module, tu: TestUtils
):
    module.forward(tu.rand(1, 512, 15, 14))


class AdaptiveAvgPool2dFixedKernelStrideSizeStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.aap2d = torch.nn.AdaptiveAvgPool2d((2, 2))

    @export
    @annotate_args(
        [
            None,
            ([1, 3, 7, 7], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.aap2d(x)


@register_test_case(
    module_factory=lambda: AdaptiveAvgPool2dFixedKernelStrideSizeStaticModule()
)
def AdaptiveAvgPool2dFixedKernelStrideSizeStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 3, 7, 7))


class AdaptiveAvgPool2dUnitOutputSizeStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.aap2d = torch.nn.AdaptiveAvgPool2d((1, 1))

    @export
    @annotate_args(
        [
            None,
            ([1, 512, 7, 7], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.aap2d(x)


@register_test_case(
    module_factory=lambda: AdaptiveAvgPool2dUnitOutputSizeStaticModule()
)
def AdaptiveAvgPool2dUnitOutputSizeStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 512, 7, 7))


class AdaptiveAvgPool2dUnitOutputSizeDynamicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.aap2d = torch.nn.AdaptiveAvgPool2d((1, 1))

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.aap2d(x)


@register_test_case(
    module_factory=lambda: AdaptiveAvgPool2dUnitOutputSizeDynamicModule()
)
def AdaptiveAvgPool2dUnitOutputSizeDynamicModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 512, 7, 7))


# ==============================================================================


class MaxPool1dModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.mp1d = torch.nn.MaxPool1d(
            kernel_size=[6], stride=[2], padding=[3], dilation=2
        )

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.mp1d(x)


@register_test_case(module_factory=lambda: MaxPool1dModule())
def MaxPool1dModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 20, low=-1))


class MaxPool1dEmptyStrideStaticModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1, 1, 20], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.max_pool1d(x, kernel_size=2, stride=[])


@register_test_case(module_factory=lambda: MaxPool1dEmptyStrideStaticModule())
def MaxPool1dEmptyStrideStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 20, low=-1))


class MaxPool1dStaticModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.mp1d = torch.nn.MaxPool1d(
            kernel_size=[3], stride=[2], padding=[1], dilation=[1]
        )

    @export
    @annotate_args(
        [
            None,
            ([1, 64, 112], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.mp1d(x)


@register_test_case(module_factory=lambda: MaxPool1dStaticModule())
def MaxPool1dStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 64, 112))


class MaxPool1dStaticCeilModeTrueModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.mp1d = torch.nn.MaxPool1d(
            kernel_size=[3], stride=[2], padding=[1], dilation=[1], ceil_mode=True
        )

    @export
    @annotate_args(
        [
            None,
            ([1, 64, 112], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.mp1d(x)


@register_test_case(module_factory=lambda: MaxPool1dStaticCeilModeTrueModule())
def MaxPool1dStaticCeilModeTrueModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 64, 112))


class MaxPool1dCeilModeTrueModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.mp1d = torch.nn.MaxPool1d(
            kernel_size=[6], stride=[2], padding=[3], dilation=2, ceil_mode=True
        )

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.mp1d(x)


@register_test_case(module_factory=lambda: MaxPool1dCeilModeTrueModule())
def MaxPool1dCeilModeTrueModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 20, low=0.5, high=1.0))


# ==============================================================================


class MaxPool2dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mp2d = torch.nn.MaxPool2d(
            kernel_size=[6, 8], stride=[2, 2], padding=[3, 4], dilation=2
        )

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.mp2d(x)


@register_test_case(module_factory=lambda: MaxPool2dModule())
def MaxPool2dModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 20, 20, low=-1))


class MaxPool2dEmptyStrideStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1, 1, 20, 20], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.max_pool2d(x, kernel_size=2, stride=[])


@register_test_case(module_factory=lambda: MaxPool2dEmptyStrideStaticModule())
def MaxPool2dEmptyStrideStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 20, 20, low=-1))


class MaxPool2dStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mp2d = torch.nn.MaxPool2d(
            kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1]
        )

    @export
    @annotate_args(
        [
            None,
            ([1, 64, 112, 112], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.mp2d(x)


@register_test_case(module_factory=lambda: MaxPool2dStaticModule())
def MaxPool2dStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 64, 112, 112))


class MaxPool2dStaticCeilModeTrueModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mp2d = torch.nn.MaxPool2d(
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            ceil_mode=True,
        )

    @export
    @annotate_args(
        [
            None,
            ([1, 64, 112, 112], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.mp2d(x)


@register_test_case(module_factory=lambda: MaxPool2dStaticCeilModeTrueModule())
def MaxPool2dStaticCeilModeTrueModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 64, 112, 112))


class MaxPool2dCeilModeTrueModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mp2d = torch.nn.MaxPool2d(
            kernel_size=[6, 8],
            stride=[2, 2],
            padding=[3, 4],
            dilation=2,
            ceil_mode=True,
        )

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.mp2d(x)


@register_test_case(module_factory=lambda: MaxPool2dCeilModeTrueModule())
def MaxPool2dCeilModeTrueModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 20, 20, low=0.5, high=1.0))


class MaxPool2dStaticCeilModeTrueReduceOutputModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mp2d = torch.nn.MaxPool2d(
            kernel_size=6,
            stride=6,
            padding=3,
            dilation=1,
            ceil_mode=True,
        )

    @export
    @annotate_args(
        [
            None,
            ([2, 6, 20, 10], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.mp2d(x)


@register_test_case(
    module_factory=lambda: MaxPool2dStaticCeilModeTrueReduceOutputModule()
)
def MaxPool2dStaticCeilModeTrueReduceOutputModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 6, 20, 10, low=0.5, high=1.0))


# ==============================================================================


class MaxPool3dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mp3d = torch.nn.MaxPool3d(
            kernel_size=[4, 4, 4], stride=[2, 2, 2], padding=[1, 1, 1], dilation=1
        )

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.mp3d(x)


@register_test_case(module_factory=lambda: MaxPool3dModule())
def MaxPool3dModule_basic(module, tu: TestUtils):
    module.forward(torch.arange(8 * 8 * 8).view(1, 1, 8, 8, 8).float())


class MaxPool3dRandomSimpleModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mp3d = torch.nn.MaxPool3d(
            kernel_size=[4, 4, 4], stride=[2, 2, 2], padding=[1, 1, 1], dilation=1
        )

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.mp3d(x)


@register_test_case(module_factory=lambda: MaxPool3dRandomSimpleModule())
def MaxPool3dModuleRandomSimple_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 20, 20, 20, low=-1))


class MaxPool3dLargeDataModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mp3d = torch.nn.MaxPool3d(
            kernel_size=[6, 8, 8], stride=[2, 2, 2], padding=[3, 4, 4], dilation=2
        )

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.mp3d(x)


@register_test_case(module_factory=lambda: MaxPool3dLargeDataModule())
def MaxPool3dLargeDatadModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 20, 20, 20, low=-1))


class MaxPool3dEmptyStrideStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1, 1, 20, 20, 20], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.max_pool3d(x, kernel_size=2, stride=[])


@register_test_case(module_factory=lambda: MaxPool3dEmptyStrideStaticModule())
def MaxPool3dEmptyStrideStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 20, 20, 20, low=-1))


class MaxPool3dStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mp3d = torch.nn.MaxPool3d(
            kernel_size=[3, 3, 3],
            stride=[2, 2, 2],
            padding=[1, 1, 1],
            dilation=[1, 1, 1],
        )

    @export
    @annotate_args(
        [
            None,
            ([1, 64, 112, 112, 112], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.mp3d(x)


@register_test_case(module_factory=lambda: MaxPool3dStaticModule())
def MaxPool3dStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 64, 112, 112, 112))


class MaxPool3dStaticCeilModeTrueModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mp3d = torch.nn.MaxPool3d(
            kernel_size=[3, 3, 3],
            stride=[2, 2, 2],
            padding=[1, 1, 1],
            dilation=[1, 1, 1],
            ceil_mode=True,
        )

    @export
    @annotate_args(
        [
            None,
            ([1, 64, 112, 112, 112], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.mp3d(x)


@register_test_case(module_factory=lambda: MaxPool3dStaticCeilModeTrueModule())
def MaxPool3dStaticCeilModeTrueModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 64, 112, 112, 112))


class MaxPool3dCeilModeTrueModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mp3d = torch.nn.MaxPool3d(
            kernel_size=[6, 8, 8],
            stride=[2, 2, 2],
            padding=[3, 4, 4],
            dilation=2,
            ceil_mode=True,
        )

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.mp3d(x)


@register_test_case(module_factory=lambda: MaxPool3dCeilModeTrueModule())
def MaxPool3dCeilModeTrueModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 20, 20, 20, low=0.5, high=1.0))


# ==============================================================================


class MaxPool2dWithIndicesModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.max_pool2d_with_indices(
            x, kernel_size=[2, 2], stride=[1, 1], padding=[0, 0], dilation=[1, 1]
        )


@register_test_case(module_factory=lambda: MaxPool2dWithIndicesModule())
def MaxPool2dWithIndicesModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 8, 8, low=0.5, high=1.0))


class MaxPool2dWithIndicesFullSizeKernelModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.max_pool2d_with_indices(
            x, kernel_size=[4, 4], stride=1, padding=0, dilation=1
        )


@register_test_case(module_factory=lambda: MaxPool2dWithIndicesFullSizeKernelModule())
def MaxPool2dWithIndicesFullSizeKernelModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4, 4, low=0.5, high=1.0))


class MaxPool2dWithIndicesNonDefaultPaddingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.max_pool2d_with_indices(
            x, kernel_size=[4, 8], stride=[1, 1], padding=[2, 4], dilation=1
        )


@register_test_case(
    module_factory=lambda: MaxPool2dWithIndicesNonDefaultPaddingModule()
)
def MaxPool2dWithIndicesNonDefaultPaddingModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4, 16, 16, low=-1.5, high=1.0))


class MaxPool2dWithIndicesNonDefaultStrideModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.max_pool2d_with_indices(
            x, kernel_size=[4, 4], stride=[1, 2], padding=0, dilation=1
        )


@register_test_case(module_factory=lambda: MaxPool2dWithIndicesNonDefaultStrideModule())
def MaxPool2dWithIndicesNonDefaultStrideModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 4, 16, 80, low=0.5, high=2.0))


class MaxPool2dWithIndicesNonDefaultDilationModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.max_pool2d_with_indices(
            x, kernel_size=[4, 4], stride=[1, 1], padding=0, dilation=[2, 2]
        )


@register_test_case(
    module_factory=lambda: MaxPool2dWithIndicesNonDefaultDilationModule()
)
def MaxPool2dWithIndicesNonDefaultDilationModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 4, 16, 80, low=0.5, high=2.0))


class MaxPool2dWithIndicesNonDefaultParamsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.max_pool2d_with_indices(
            x, kernel_size=[8, 4], stride=[2, 2], padding=[1, 2], dilation=[2, 2]
        )


@register_test_case(module_factory=lambda: MaxPool2dWithIndicesNonDefaultParamsModule())
def MaxPool2dWithIndicesNonDefaultParamsModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 4, 16, 80, low=-0.5, high=4.0))


class MaxPool2dWithIndicesAllNegativeValuesModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.max_pool2d_with_indices(
            x, kernel_size=[4, 8], stride=[1, 1], padding=[2, 4], dilation=1
        )


@register_test_case(
    module_factory=lambda: MaxPool2dWithIndicesAllNegativeValuesModule()
)
def MaxPool2dWithIndicesAllNegativeValuesModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4, 16, 16, low=-4.5, high=-1.0))


class MaxPool2dWithIndicesStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 4, 16, 16], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.max_pool2d_with_indices(
            x, kernel_size=[4, 8], stride=[1, 1], padding=[2, 4], dilation=1
        )


@register_test_case(module_factory=lambda: MaxPool2dWithIndicesStaticModule())
def MaxPool2dWithIndicesStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4, 16, 16, low=-4.5, high=-1.0))


class MaxPool2dWithIndicesAllOnesModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.max_pool2d_with_indices(
            x, kernel_size=[2, 2], stride=[1, 1], padding=[0, 0], dilation=[1, 1]
        )


@register_test_case(module_factory=lambda: MaxPool2dWithIndicesAllOnesModule())
def MaxPool2dWithIndicesAllOnesModule_basic(module, tu: TestUtils):
    module.forward(torch.ones(1, 1, 8, 8))


class MaxPool2dWithIndicesCeilModeTrueModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.max_pool2d_with_indices(
            x,
            kernel_size=[2, 2],
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            ceil_mode=True,
        )


@register_test_case(module_factory=lambda: MaxPool2dWithIndicesCeilModeTrueModule())
def MaxPool2dWithIndicesCeilModeTrueModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 8, 8, low=0.5, high=1.0))


# ==============================================================================


class MaxPool2dWithIndicesBackwardStatic4DModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 4, 7, 6], torch.float32, True),
            ([2, 4, 6, 5], torch.float32, True),
            ([2, 4, 7, 6], torch.int64, True),
        ]
    )
    def forward(self, output, input, indices):
        kernel_size = [2, 2]
        stride = [1, 1]
        padding = [1, 1]
        dilation = [1, 1]
        ceil_mode = False
        return torch.ops.aten.max_pool2d_with_indices_backward(
            output, input, kernel_size, stride, padding, dilation, ceil_mode, indices
        )


@register_test_case(module_factory=lambda: MaxPool2dWithIndicesBackwardStatic4DModule())
def MaxPool2dWithIndicesBackwardStatic4DModule_basic(module, tu: TestUtils):
    module.forward(
        tu.rand(2, 4, 7, 6), tu.rand(2, 4, 6, 5), tu.randint(2, 4, 7, 6, high=16)
    )


class MaxPool2dWithIndicesBackwardStatic3DModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([4, 7, 6], torch.float32, True),
            ([4, 6, 5], torch.float32, True),
            ([4, 7, 6], torch.int64, True),
        ]
    )
    def forward(self, output, input, indices):
        kernel_size = [2, 2]
        stride = [1, 1]
        padding = [1, 1]
        dilation = [1, 1]
        ceil_mode = False
        return torch.ops.aten.max_pool2d_with_indices_backward(
            output, input, kernel_size, stride, padding, dilation, ceil_mode, indices
        )


@register_test_case(module_factory=lambda: MaxPool2dWithIndicesBackwardStatic3DModule())
def MaxPool2dWithIndicesBackwardStatic3DModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 7, 6), tu.rand(4, 6, 5), tu.randint(4, 7, 6, high=16))


class MaxPool2dWithIndicesBackwardDynamic4DModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
            ([-1, -1, -1, -1], torch.float32, True),
            ([-1, -1, -1, -1], torch.int64, True),
        ]
    )
    def forward(self, output, input, indices):
        kernel_size = [2, 2]
        stride = [1, 1]
        padding = [1, 1]
        dilation = [1, 1]
        ceil_mode = False
        return torch.ops.aten.max_pool2d_with_indices_backward(
            output, input, kernel_size, stride, padding, dilation, ceil_mode, indices
        )


@register_test_case(
    module_factory=lambda: MaxPool2dWithIndicesBackwardDynamic4DModule()
)
def MaxPool2dWithIndicesBackwardDynamic4DModule_basic(module, tu: TestUtils):
    module.forward(
        tu.rand(2, 4, 7, 6), tu.rand(2, 4, 6, 5), tu.randint(2, 4, 7, 6, high=16)
    )


class MaxPool2dWithIndicesBackwardDynamic3DModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1, -1], torch.int64, True),
        ]
    )
    def forward(self, output, input, indices):
        kernel_size = [2, 2]
        stride = [1, 1]
        padding = [1, 1]
        dilation = [1, 1]
        ceil_mode = False
        return torch.ops.aten.max_pool2d_with_indices_backward(
            output, input, kernel_size, stride, padding, dilation, ceil_mode, indices
        )


@register_test_case(
    module_factory=lambda: MaxPool2dWithIndicesBackwardDynamic3DModule()
)
def MaxPool2dWithIndicesBackwardDynamic3DModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 7, 6), tu.rand(2, 6, 5), tu.randint(2, 7, 6, high=16))


# ==============================================================================


class MaxPool3dWithIndicesModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.max_pool3d_with_indices(
            x,
            kernel_size=[2, 2, 2],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            dilation=[1, 1, 1],
        )


@register_test_case(module_factory=lambda: MaxPool3dWithIndicesModule())
def MaxPool3dWithIndicesModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 8, 8, 8, low=0.5, high=1.0))


class MaxPool3dWithIndicesFullSizeKernelModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.max_pool3d_with_indices(
            x, kernel_size=[4, 4, 4], stride=1, padding=0, dilation=1
        )


@register_test_case(module_factory=lambda: MaxPool3dWithIndicesFullSizeKernelModule())
def MaxPool3dWithIndicesFullSizeKernelModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4, 4, 4, low=0.5, high=1.0))


class MaxPool3dWithIndicesNonDefaultPaddingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.max_pool3d_with_indices(
            x, kernel_size=[4, 8, 4], stride=[1, 1, 1], padding=[2, 4, 2], dilation=1
        )


@register_test_case(
    module_factory=lambda: MaxPool3dWithIndicesNonDefaultPaddingModule()
)
def MaxPool3dWithIndicesNonDefaultPaddingModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4, 16, 16, 16, low=-1.5, high=1.0))


class MaxPool3dWithIndicesNonDefaultStrideModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.max_pool3d_with_indices(
            x, kernel_size=[4, 4, 4], stride=[1, 2, 1], padding=0, dilation=1
        )


@register_test_case(module_factory=lambda: MaxPool3dWithIndicesNonDefaultStrideModule())
def MaxPool3dWithIndicesNonDefaultStrideModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 4, 16, 80, 16, low=0.5, high=2.0))


class MaxPool3dWithIndicesNonDefaultDilationModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.max_pool3d_with_indices(
            x, kernel_size=[4, 4, 4], stride=[1, 1, 1], padding=0, dilation=[2, 2, 2]
        )


@register_test_case(
    module_factory=lambda: MaxPool3dWithIndicesNonDefaultDilationModule()
)
def MaxPool3dWithIndicesNonDefaultDilationModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 4, 16, 80, 16, low=0.5, high=2.0))


class MaxPool3dWithIndicesNonDefaultParamsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.max_pool3d_with_indices(
            x,
            kernel_size=[8, 4, 8],
            stride=[2, 2, 2],
            padding=[1, 2, 1],
            dilation=[2, 2, 2],
        )


@register_test_case(module_factory=lambda: MaxPool3dWithIndicesNonDefaultParamsModule())
def MaxPool3dWithIndicesNonDefaultParamsModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 4, 16, 80, 16, low=-0.5, high=4.0))


class MaxPool3dWithIndicesAllNegativeValuesModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.max_pool3d_with_indices(
            x, kernel_size=[4, 8, 4], stride=[1, 1, 1], padding=[2, 4, 2], dilation=1
        )


@register_test_case(
    module_factory=lambda: MaxPool3dWithIndicesAllNegativeValuesModule()
)
def MaxPool3dWithIndicesAllNegativeValuesModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4, 16, 16, 16, low=-4.5, high=-1.0))


class MaxPool3dWithIndicesStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 4, 16, 16, 16], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.max_pool3d_with_indices(
            x, kernel_size=[4, 8, 4], stride=[1, 1, 1], padding=[2, 4, 2], dilation=1
        )


@register_test_case(module_factory=lambda: MaxPool3dWithIndicesStaticModule())
def MaxPool3dWithIndicesStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4, 16, 16, 16, low=-4.5, high=-1.0))


class MaxPool3dWithIndicesAllOnesModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.max_pool3d_with_indices(
            x,
            kernel_size=[2, 2, 2],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            dilation=[1, 1, 1],
        )


@register_test_case(module_factory=lambda: MaxPool3dWithIndicesAllOnesModule())
def MaxPool3dWithIndicesAllOnesModule_basic(module, tu: TestUtils):
    module.forward(torch.ones(1, 1, 8, 8, 8))


class MaxPool3dWithIndicesCeilModeTrueModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.max_pool3d_with_indices(
            x,
            kernel_size=[2, 2, 2],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            dilation=[1, 1, 1],
            ceil_mode=True,
        )


@register_test_case(module_factory=lambda: MaxPool3dWithIndicesCeilModeTrueModule())
def MaxPool3dWithIndicesCeilModeTrueModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 8, 8, 8, low=0.5, high=1.0))


# ==============================================================================


class AvgPool2dFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ap2d = torch.nn.AvgPool2d(
            kernel_size=[6, 8],
            stride=[2, 2],
            padding=[3, 4],
            ceil_mode=False,
            count_include_pad=True,
            divisor_override=None,
        )

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.ap2d(x)


@register_test_case(module_factory=lambda: AvgPool2dFloatModule())
def AvgPool2dFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4, 20, 20, low=-1))


class AvgPool2dIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ap2d = torch.nn.AvgPool2d(
            kernel_size=[6, 8],
            stride=[2, 2],
            padding=[3, 4],
            ceil_mode=False,
            count_include_pad=True,
            divisor_override=None,
        )

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.int64, True),
        ]
    )
    def forward(self, x):
        return self.ap2d(x)


@register_test_case(module_factory=lambda: AvgPool2dIntModule())
def AvgPool2dIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(2, 4, 20, 20, high=100))


class AvgPool2dStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ap2d = torch.nn.AvgPool2d(
            kernel_size=[6, 8],
            stride=[2, 2],
            padding=[3, 4],
            ceil_mode=False,
            count_include_pad=True,
            divisor_override=None,
        )

    @export
    @annotate_args(
        [
            None,
            ([2, 2, 10, 20], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.ap2d(x)


@register_test_case(module_factory=lambda: AvgPool2dStaticModule())
def AvgPool2dStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 2, 10, 20, low=-1))


class AvgPool2dCountIncludePadFalseStaticModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.ap2d = torch.nn.AvgPool2d(
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1],
            ceil_mode=False,
            count_include_pad=False,
            divisor_override=None,
        )

    @export
    @annotate_args(
        [
            None,
            ([32, 384, 25, 25], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.ap2d(x)


@register_test_case(module_factory=lambda: AvgPool2dCountIncludePadFalseStaticModule())
def AvgPool2dCountIncludePadFalseStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(32, 384, 25, 25, low=-1))


class AvgPool2dDivisorOverrideModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ap2d = torch.nn.AvgPool2d(
            kernel_size=[4, 8],
            stride=[2, 3],
            padding=[2, 4],
            ceil_mode=False,
            count_include_pad=True,
            divisor_override=22,
        )

    @export
    @annotate_args(
        [
            None,
            ([4, 4, 20, 20], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.ap2d(x)


@register_test_case(module_factory=lambda: AvgPool2dDivisorOverrideModule())
def AvgPool2dDivisorOverrideModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 4, 20, 20, low=-1))


class AvgPool2dCeilModeTrueModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ap2d = torch.nn.AvgPool2d(
            kernel_size=[6, 8],
            stride=[2, 2],
            padding=[3, 4],
            ceil_mode=False,
            count_include_pad=True,
            divisor_override=None,
        )

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.ap2d(x)


@register_test_case(module_factory=lambda: AvgPool2dCeilModeTrueModule())
def AvgPool2dCeilModeTrueModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4, 20, 20, low=0.5, high=1.0))


class AvgPool2dWithoutPadModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ap2d = torch.nn.AvgPool2d(
            kernel_size=[6, 8],
            stride=[2, 2],
            padding=[0, 0],
            ceil_mode=False,
            count_include_pad=False,
            divisor_override=None,
        )

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.ap2d(x)


@register_test_case(module_factory=lambda: AvgPool2dWithoutPadModule())
def AvgPool2dWithoutPadModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4, 20, 20, low=0.5, high=1.0))


class AvgPool2dCHWModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ap2d = torch.nn.AvgPool2d(
            kernel_size=[6, 8],
            stride=[2, 2],
        )

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.ap2d(x)


@register_test_case(module_factory=lambda: AvgPool2dCHWModule())
def AvgPool2dCHWModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 20, 20, low=0.5, high=1.0))


class AvgPool2dSingleIntTupleParamsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ap2d = torch.nn.AvgPool2d(
            kernel_size=(6,),
            stride=(2,),
            padding=(1,),
            count_include_pad=False,
        )

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.ap2d(x)


@register_test_case(module_factory=lambda: AvgPool2dSingleIntTupleParamsModule())
def AvgPool2dSingleIntTupleParamsModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4, 20, 20, low=0.5, high=1.0))


class AvgPool2dSingleIntTupleParamsIncludePadModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ap2d = torch.nn.AvgPool2d(
            kernel_size=(6,),
            stride=(2,),
            padding=(1,),
            count_include_pad=True,
        )

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.ap2d(x)


@register_test_case(
    module_factory=lambda: AvgPool2dSingleIntTupleParamsIncludePadModule()
)
def AvgPool2dSingleIntTupleParamsIncludePadModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4, 20, 20, low=0.5, high=1.0))


# ==============================================================================


class AvgPool3dStaticModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.ap2d = torch.nn.AvgPool3d(
            kernel_size=[2, 2, 2],
            stride=[2, 2, 2],
            padding=[0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            divisor_override=None,
        )

    @export
    @annotate_args(
        [
            None,
            ([2, 2, 4, 4, 4], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.ap2d(x)


@register_test_case(module_factory=lambda: AvgPool3dStaticModule())
def AvgPool3dStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 2, 4, 4, 4, low=-1))


class AvgPool3dCountIncludePadFalse(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.ap3d = torch.nn.AvgPool3d(
            kernel_size=[3, 3, 3],
            stride=[1, 1, 1],
            padding=[1, 1, 1],
            ceil_mode=False,
            count_include_pad=False,
            divisor_override=None,
        )

    @export
    @annotate_args(
        [
            None,
            ([3, 3, 12, 12, 12], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.ap3d(x)


@register_test_case(module_factory=lambda: AvgPool3dCountIncludePadFalse())
def AvgPool3dCountIncludePadFalse_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 3, 12, 12, 12, low=-1))


class AvgPool3dCountIncludePadFalseWithoutPadding(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.ap3d = torch.nn.AvgPool3d(
            kernel_size=[3, 3, 3],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            ceil_mode=False,
            count_include_pad=False,
            divisor_override=None,
        )

    @export
    @annotate_args(
        [
            None,
            ([3, 3, 12, 12, 12], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.ap3d(x)


@register_test_case(
    module_factory=lambda: AvgPool3dCountIncludePadFalseWithoutPadding()
)
def AvgPool3dCountIncludePadFalseWithoutPadding_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 3, 12, 12, 12, low=-1))


# ==============================================================================


class AvgPool1dFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ap1d = torch.nn.AvgPool1d(
            kernel_size=6, stride=2, padding=3, ceil_mode=False, count_include_pad=True
        )

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.ap1d(x)


@register_test_case(module_factory=lambda: AvgPool1dFloatModule())
def AvgPool1dFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4, 20, low=-1))


class AvgPool1dIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ap1d = torch.nn.AvgPool1d(
            kernel_size=6, stride=2, padding=3, ceil_mode=False, count_include_pad=True
        )

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.int64, True),
        ]
    )
    def forward(self, x):
        return self.ap1d(x)


@register_test_case(module_factory=lambda: AvgPool1dIntModule())
def AvgPool1dIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(2, 4, 20, high=100))


class AvgPool1dStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ap1d = torch.nn.AvgPool1d(
            kernel_size=6, stride=2, padding=3, ceil_mode=False, count_include_pad=True
        )

    @export
    @annotate_args(
        [
            None,
            ([2, 4, 20], torch.int64, True),
        ]
    )
    def forward(self, x):
        return self.ap1d(x)


@register_test_case(module_factory=lambda: AvgPool1dStaticModule())
def AvgPool1dStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(2, 4, 20, high=100))


class AvgPool1dCountIncludePadFalseWithoutPadding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ap1d = torch.nn.AvgPool1d(
            kernel_size=3, stride=1, padding=0, ceil_mode=False, count_include_pad=False
        )

    @export
    @annotate_args(
        [
            None,
            ([3, 4, 20], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.ap1d(x)


@register_test_case(
    module_factory=lambda: AvgPool1dCountIncludePadFalseWithoutPadding()
)
def AvgPool1dCountIncludePadFalseWithoutPadding_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 20))


class AvgPool1dCountIncludePadFalse(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ap1d = torch.nn.AvgPool1d(
            kernel_size=3, stride=1, padding=1, ceil_mode=False, count_include_pad=False
        )

    @export
    @annotate_args(
        [
            None,
            ([3, 4, 20], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.ap1d(x)


@register_test_case(module_factory=lambda: AvgPool1dCountIncludePadFalse())
def AvgPool1dCountIncludePadFalse_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 20))


# ==============================================================================


class AdaptiveAvgPool1dStaticLargerOutput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.aap1d = torch.nn.AdaptiveAvgPool1d(output_size=13)

    @export
    @annotate_args([None, ([5, 512, 7], torch.float32, True)])
    def forward(self, x):
        return self.aap1d(x)


@register_test_case(module_factory=lambda: AdaptiveAvgPool1dStaticLargerOutput())
def AdaptiveAvgPool1dStaticLargerOutput_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 512, 7))


class AdaptiveAvgPool1dStaticEvenMultiple(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.aap1d = torch.nn.AdaptiveAvgPool1d(output_size=7)

    @export
    @annotate_args([None, ([5, 512, 147], torch.float32, True)])
    def forward(self, x):
        return self.aap1d(x)


@register_test_case(module_factory=lambda: AdaptiveAvgPool1dStaticEvenMultiple())
def AdaptiveAvgPool1dStaticEvenMultiple_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 512, 147))


class AdaptiveAvgPool1dGeneralDynamic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.aap1d = torch.nn.AdaptiveAvgPool1d(output_size=7)

    @export
    @annotate_args([None, ([-1, -1, -1], torch.float32, True)])
    def forward(self, x):
        return self.aap1d(x)


@register_test_case(module_factory=lambda: AdaptiveAvgPool1dGeneralDynamic())
def AdaptiveAvgPool1dGeneralDynamic_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 512, 10))


class AdaptiveAvgPool1dGeneralDynamicNoBatches(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.aap1d = torch.nn.AdaptiveAvgPool1d(output_size=7)

    @export
    @annotate_args([None, ([-1, -1], torch.float32, True)])
    def forward(self, x):
        return self.aap1d(x)


@register_test_case(module_factory=lambda: AdaptiveAvgPool1dGeneralDynamicNoBatches())
def AdaptiveAvgPool1dGeneralDynamicNoBatches_basic(module, tu: TestUtils):
    module.forward(tu.rand(512, 10))


class AdaptiveAvgPool1dNonUnitOutputSizeStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.aap1d = torch.nn.AdaptiveAvgPool1d(output_size=7)

    @export
    @annotate_args(
        [
            None,
            ([1, 512, 7], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.aap1d(x)


@register_test_case(
    module_factory=lambda: AdaptiveAvgPool1dNonUnitOutputSizeStaticModule()
)
def AdaptiveAvgPool1dNonUnitOutputSizeStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 512, 7))


class AdaptiveAvgPool1dNonUnitOutputSizeDynamicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.aap1d = torch.nn.AdaptiveAvgPool1d(output_size=7)

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.aap1d(x)


@register_test_case(
    module_factory=lambda: AdaptiveAvgPool1dNonUnitOutputSizeDynamicModule()
)
def AdaptiveAvgPool1dNonUnitOutputSizeDynamicModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 512, 7))


class AdaptiveAvgPool1dUnitOutputSizeStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.aap1d = torch.nn.AdaptiveAvgPool1d(output_size=1)

    @export
    @annotate_args(
        [
            None,
            ([1, 512, 7], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.aap1d(x)


@register_test_case(
    module_factory=lambda: AdaptiveAvgPool1dUnitOutputSizeStaticModule()
)
def AdaptiveAvgPool1dUnitOutputSizeStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 512, 7))


class AdaptiveAvgPool1dUnitOutputSizeDynamicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.aap1d = torch.nn.AdaptiveAvgPool1d(output_size=1)

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.aap1d(x)


@register_test_case(
    module_factory=lambda: AdaptiveAvgPool1dUnitOutputSizeDynamicModule()
)
def AdaptiveAvgPool1dUnitOutputSizeDynamicModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 512, 7))


# AdaptiveAvgPool2d


class AdaptiveAvgPool2dDynamic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.aap2d = torch.nn.AdaptiveAvgPool2d(output_size=(7, 13))

    @export
    @annotate_args([None, ([-1, -1, -1, -1], torch.float32, True)])
    def forward(self, x):
        return self.aap2d(x)


@register_test_case(module_factory=lambda: AdaptiveAvgPool2dDynamic())
def AdaptiveAvgPool2dDynamic_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 512, 10, 16))


class AdaptiveAvgPool2dDynamicNoBatch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.aap2d = torch.nn.AdaptiveAvgPool2d(output_size=(7, 13))

    @export
    @annotate_args([None, ([-1, -1, -1], torch.float32, True)])
    def forward(self, x):
        return self.aap2d(x)


@register_test_case(module_factory=lambda: AdaptiveAvgPool2dDynamicNoBatch())
def AdaptiveAvgPool2dDynamicNoBatch_basic(module, tu: TestUtils):
    module.forward(tu.rand(512, 10, 16))


# AdaptiveAvgPool3d


class AdaptiveAvgPool3dDynamic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.aap3d = torch.nn.AdaptiveAvgPool3d(output_size=(7, 13, 15))

    @export
    @annotate_args([None, ([-1, -1, -1, -1, -1], torch.float32, True)])
    def forward(self, x):
        return self.aap3d(x)


@register_test_case(module_factory=lambda: AdaptiveAvgPool3dDynamic())
def AdaptiveAvgPool3dDynamic_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 512, 10, 16, 17))


class AdaptiveAvgPool3dDynamicNoBatch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.aap3d = torch.nn.AdaptiveAvgPool3d(output_size=(7, 13, 15))

    @export
    @annotate_args([None, ([-1, -1, -1, -1], torch.float32, True)])
    def forward(self, x):
        return self.aap3d(x)


@register_test_case(module_factory=lambda: AdaptiveAvgPool3dDynamicNoBatch())
def AdaptiveAvgPool3dDynamicNoBatch_basic(module, tu: TestUtils):
    module.forward(tu.rand(512, 10, 16, 17))


# AdaptiveMaxPool1d


class AdaptiveMaxPool1dDynamic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.amp1d = torch.nn.AdaptiveMaxPool1d(output_size=(7), return_indices=False)

    @export
    @annotate_args([None, ([-1, -1, -1], torch.float32, True)])
    def forward(self, x):
        return self.amp1d(x)


@register_test_case(module_factory=lambda: AdaptiveMaxPool1dDynamic())
def AdaptiveMaxPool1dDynamic_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 512, 10))


class AdaptiveMaxPool1dDynamicNoBatch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.amp1d = torch.nn.AdaptiveMaxPool1d(output_size=(7), return_indices=False)

    @export
    @annotate_args([None, ([-1, -1], torch.float32, True)])
    def forward(self, x):
        return self.amp1d(x)


@register_test_case(module_factory=lambda: AdaptiveMaxPool1dDynamicNoBatch())
def AdaptiveMaxPool1dDynamicNoBatch_basic(module, tu: TestUtils):
    module.forward(tu.rand(512, 10))


class AdaptiveMaxPool1dStatic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.amp1d = torch.nn.AdaptiveMaxPool1d(output_size=(7), return_indices=False)

    @export
    @annotate_args([None, ([1, 512, 10], torch.float32, True)])
    def forward(self, x):
        return self.amp1d(x)


@register_test_case(module_factory=lambda: AdaptiveMaxPool1dStatic())
def AdaptiveMaxPool1dStatic_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 512, 10))


class AdaptiveMaxPool1dDimOneStatic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.amp1d = torch.nn.AdaptiveMaxPool1d(output_size=(1), return_indices=False)

    @export
    @annotate_args([None, ([1, 512, 7], torch.float32, True)])
    def forward(self, x):
        return self.amp1d(x)


@register_test_case(module_factory=lambda: AdaptiveMaxPool1dDimOneStatic())
def AdaptiveMaxPool1dDimOneStatic_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 512, 7))


# AdaptiveMaxPool2d


class AdaptiveMaxPool2dDynamic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.amp2d = torch.nn.AdaptiveMaxPool2d(
            output_size=(7, 13), return_indices=False
        )

    @export
    @annotate_args([None, ([-1, -1, -1, -1], torch.float32, True)])
    def forward(self, x):
        return self.amp2d(x)


@register_test_case(module_factory=lambda: AdaptiveMaxPool2dDynamic())
def AdaptiveMaxPool2dDynamic_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 512, 10, 16))


class AdaptiveMaxPool2dDynamicNoBatch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.amp2d = torch.nn.AdaptiveMaxPool2d(
            output_size=(7, 13), return_indices=False
        )

    @export
    @annotate_args([None, ([-1, -1, -1], torch.float32, True)])
    def forward(self, x):
        return self.amp2d(x)


@register_test_case(module_factory=lambda: AdaptiveMaxPool2dDynamicNoBatch())
def AdaptiveMaxPool2dDynamicNoBatch_basic(module, tu: TestUtils):
    module.forward(tu.rand(512, 10, 16))


class AdaptiveMaxPool2dDynamicWithIndices(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.amp2d = torch.nn.AdaptiveMaxPool2d(
            output_size=(7, 13), return_indices=True
        )

    @export
    @annotate_args([None, ([-1, -1, -1, -1], torch.float32, True)])
    def forward(self, x):
        return self.amp2d(x)


@register_test_case(module_factory=lambda: AdaptiveMaxPool2dDynamicWithIndices())
def AdaptiveMaxPool2dDynamicWithIndices_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 512, 10, 16))


class AdaptiveMaxPool2dStatic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.amp2d = torch.nn.AdaptiveMaxPool2d(
            output_size=(7, 13), return_indices=False
        )

    @export
    @annotate_args([None, ([1, 512, 10, 9], torch.float32, True)])
    def forward(self, x):
        return self.amp2d(x)


@register_test_case(module_factory=lambda: AdaptiveMaxPool2dStatic())
def AdaptiveMaxPool2dStatic_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 512, 10, 9))


class AdaptiveMaxPool2dStaticWithIndices(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.amp2d = torch.nn.AdaptiveMaxPool2d(
            output_size=(7, 13), return_indices=True
        )

    @export
    @annotate_args([None, ([1, 512, 10, 16], torch.float32, True)])
    def forward(self, x):
        return self.amp2d(x)


@register_test_case(module_factory=lambda: AdaptiveMaxPool2dStaticWithIndices())
def AdaptiveMaxPool2dStaticWithIndices_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 512, 10, 16))


class AdaptiveMaxPool2dFixedKernelStrideSizeStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.amp2d = torch.nn.AdaptiveMaxPool2d((2, 2))

    @export
    @annotate_args(
        [
            None,
            ([1, 3, 7, 7], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.amp2d(x)


@register_test_case(
    module_factory=lambda: AdaptiveMaxPool2dFixedKernelStrideSizeStaticModule()
)
def AdaptiveMaxPool2dFixedKernelStrideSizeStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 3, 7, 7))


class AdaptiveMaxPool2dUnitOutputSizeStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.amp2d = torch.nn.AdaptiveMaxPool2d((1, 1))

    @export
    @annotate_args(
        [
            None,
            ([1, 512, 7, 7], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.amp2d(x)


@register_test_case(
    module_factory=lambda: AdaptiveMaxPool2dUnitOutputSizeStaticModule()
)
def AdaptiveMaxPool2dUnitOutputSizeStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 512, 7, 7))


# AdaptiveMaxPool3d


class AdaptiveMaxPool3dDynamic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.amp3d = torch.nn.AdaptiveMaxPool3d(
            output_size=(7, 13, 15), return_indices=False
        )

    @export
    @annotate_args([None, ([-1, -1, -1, -1, -1], torch.float32, True)])
    def forward(self, x):
        return self.amp3d(x)


@register_test_case(module_factory=lambda: AdaptiveMaxPool3dDynamic())
def AdaptiveMaxPool3dDynamic_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 512, 10, 16, 17))


class AdaptiveMaxPool3dDynamicNoBatch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.amp3d = torch.nn.AdaptiveMaxPool3d(
            output_size=(7, 13, 15), return_indices=False
        )

    @export
    @annotate_args([None, ([-1, -1, -1, -1], torch.float32, True)])
    def forward(self, x):
        return self.amp3d(x)


@register_test_case(module_factory=lambda: AdaptiveMaxPool3dDynamicNoBatch())
def AdaptiveMaxPool3dDynamicNoBatch_basic(module, tu: TestUtils):
    module.forward(tu.rand(512, 10, 16, 17))


class AdaptiveMaxPool3dDynamicWithIndices(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.amp3d = torch.nn.AdaptiveMaxPool3d(
            output_size=(7, 13, 15), return_indices=True
        )

    @export
    @annotate_args([None, ([-1, -1, -1, -1, -1], torch.float32, True)])
    def forward(self, x):
        return self.amp3d(x)


@register_test_case(module_factory=lambda: AdaptiveMaxPool3dDynamicWithIndices())
def AdaptiveMaxPool3dDynamicWithIndices_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 512, 10, 16, 17))


class AdaptiveMaxPool3dStatic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.amp3d = torch.nn.AdaptiveMaxPool3d(
            output_size=(7, 13, 15), return_indices=False
        )

    @export
    @annotate_args([None, ([1, 512, 10, 9, 5], torch.float32, True)])
    def forward(self, x):
        return self.amp3d(x)


@register_test_case(module_factory=lambda: AdaptiveMaxPool3dStatic())
def AdaptiveMaxPool3dStatic_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 512, 10, 9, 5))


class AdaptiveMaxPool3dStaticWithIndices(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.amp3d = torch.nn.AdaptiveMaxPool3d(
            output_size=(7, 13, 15), return_indices=True
        )

    @export
    @annotate_args([None, ([1, 512, 10, 16, 17], torch.float32, True)])
    def forward(self, x):
        return self.amp3d(x)


@register_test_case(module_factory=lambda: AdaptiveMaxPool3dStaticWithIndices())
def AdaptiveMaxPool3dStaticWithIndices_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 512, 10, 16, 17))


# ==============================================================================


class MaxUnpool3dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, 2, 2, 4], torch.float32, True),
            ([-1, -1, 2, 2, 4], torch.int64, True),
        ]
    )
    def forward(self, x, indices):
        return torch.ops.aten.max_unpool3d(x, indices, (4, 5, 6), (2, 3, 2), (0, 0, 1))


@register_test_case(module_factory=lambda: MaxUnpool3dModule())
def MaxUnpool3dModule_basic(module, tu: TestUtils):
    input = tu.rand(2, 2, 4, 5, 6)
    pool = torch.nn.MaxPool3d(
        kernel_size=(2, 2, 2), stride=(2, 3, 2), padding=(0, 0, 1), return_indices=True
    )
    output, indices = pool(input)

    module.forward(output, indices)


# We have a special case for all-zeros padding, test it too.
class MaxUnpool3dModulePad0(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, 2, 2, 3], torch.float32, True),
            ([-1, -1, 2, 2, 3], torch.int64, True),
        ]
    )
    def forward(self, x, indices):
        return torch.ops.aten.max_unpool3d(x, indices, (4, 5, 6), (2, 3, 2), (0, 0, 0))


@register_test_case(module_factory=lambda: MaxUnpool3dModulePad0())
def MaxUnpool3dModulePad0_basic(module, tu: TestUtils):
    input = tu.rand(2, 2, 4, 5, 6)
    pool = torch.nn.MaxPool3d(
        kernel_size=(2, 2, 2), stride=(2, 3, 2), padding=(0, 0, 0), return_indices=True
    )
    output, indices = pool(input)

    module.forward(output, indices)
