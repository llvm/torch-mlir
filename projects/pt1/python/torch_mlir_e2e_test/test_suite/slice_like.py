# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================


class SliceModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return x[0:5:1, 1:3:1, 2:4:1]


@register_test_case(module_factory=lambda: SliceModule())
def SliceModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6, 4, 7))


# ==============================================================================


class SliceStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([6, 4, 7], torch.float32, True),
        ]
    )
    def forward(self, x):
        return x[0:5:1, 1:3:1, 2:4:1]


@register_test_case(module_factory=lambda: SliceStaticModule())
def SliceStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6, 4, 7))


# ==============================================================================


class SliceStaticComplexInputModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([6, 4, 7], torch.complex64, True),
        ]
    )
    def forward(self, x):
        return x[0:5:1, 1:3:1, 2:4:1]


@register_test_case(module_factory=lambda: SliceStaticComplexInputModule())
def SliceStaticComplexInputModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6, 4, 7).to(torch.complex64))


# ==============================================================================


class SliceOutOfUpperBoundIndexModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        # TODO: remove hacky cat tensor once refbackend supports 0 size dim
        result = x[:8, :5, 8:]
        cat_tensor = torch.ones((6, 4, 1), dtype=torch.float32)
        return torch.cat((result, cat_tensor), dim=2)


@register_test_case(module_factory=lambda: SliceOutOfUpperBoundIndexModule())
def SliceOutOfUpperBoundIndexModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6, 4, 7))


# ==============================================================================


class SliceOutOfUpperBoundIndexStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([6, 4, 7], torch.float32, True),
        ]
    )
    def forward(self, x):
        # TODO: remove hacky cat tensor once refbackend supports 0 size dim
        result = x[:8, :5, 8:]
        cat_tensor = torch.ones((6, 4, 1), dtype=torch.float32)
        return torch.cat((result, cat_tensor), dim=2)


@register_test_case(module_factory=lambda: SliceOutOfUpperBoundIndexStaticModule())
def SliceOutOfUpperBoundIndexStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6, 4, 7))


# ==============================================================================


class SliceOutOfLowerBoundEndIndexModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return x[:-8, -7:, :]


@register_test_case(module_factory=lambda: SliceOutOfLowerBoundEndIndexModule())
def SliceOutOfLowerBoundEndIndexModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6, 4, 7))


# ==============================================================================


class SliceOutOfLowerBoundStartIndexModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return x[-8:3:1, 1:3:1, 2:4:1]


@register_test_case(module_factory=lambda: SliceOutOfLowerBoundStartIndexModule())
def SliceOutOfLowerBoundStartIndexModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6, 4, 7))


# ==============================================================================


class SliceEndSleStartModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        # TODO: remove hacky cat tensor once refbackend supports 0 size dim
        result = x[:, 4:3, :]
        cat_tensor = torch.ones((6, 1, 7), dtype=torch.float32)
        return torch.cat((result, cat_tensor), dim=1)


@register_test_case(module_factory=lambda: SliceEndSleStartModule())
def SliceEndSleStartModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6, 4, 7))


# ==============================================================================


class SliceStartEqEndModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        # TODO: remove hacky cat tensor once refbackend supports 0 size dim
        result = x[5:5, :, :]
        cat_tensor = torch.ones((1, 4, 7), dtype=torch.float32)
        return torch.cat((result, cat_tensor), dim=0)


@register_test_case(module_factory=lambda: SliceStartEqEndModule())
def SliceStartEqEndModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6, 4, 7))


# ==============================================================================


class SliceSizeTwoStepModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return x[0:5:2, 0:3:2, 0:4:2]


@register_test_case(module_factory=lambda: SliceSizeTwoStepModule())
def SliceSizeTwoStepModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(10, 5, 17))


# ==============================================================================


class SliceNegIdxModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return x[:-1, -2:-1]


@register_test_case(module_factory=lambda: SliceNegIdxModule())
def SliceNegIdxModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 9))


# ==============================================================================


class SliceSingleIdxModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return x[0]


@register_test_case(module_factory=lambda: SliceSingleIdxModule())
def SliceSingleIdxModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6, 8))


# ==============================================================================


class SliceWholeTensorModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return x[:, :]


@register_test_case(module_factory=lambda: SliceWholeTensorModule())
def SliceWholeTensorModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6, 8))


# ==============================================================================


class SelectIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.int64, True),
        ]
    )
    def forward(self, x):
        return torch.select(x, dim=0, index=0)


@register_test_case(module_factory=lambda: SelectIntModule())
def SelectIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(5, 5, high=10))


class SelectIntNegativeDimAndIndexStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([5, 5], torch.int64, True),
        ]
    )
    def forward(self, x):
        return torch.select(x, dim=-1, index=-1)


@register_test_case(module_factory=lambda: SelectIntNegativeDimAndIndexStaticModule())
def SelectIntNegativeDimAndIndexStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(5, 5, high=10))


# ==============================================================================


# For aten.slice_scatter op, The arguments are: SliceScatter(input, src, dim=0, start=None, end=None, step=1).
# For aten.select_scatter op, The arguments are: SelectScatter(input, src, dim=0, index).
class SliceScatterModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, x, src):
        return torch.ops.aten.slice_scatter(x, src, dim=1, start=0, end=1, step=1)


@register_test_case(module_factory=lambda: SliceScatterModule())
def SliceScatterModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6, 8), tu.rand(6, 1))


class SliceScatterZeroDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, x, src):
        return torch.ops.aten.slice_scatter(x, src, dim=0, start=0, end=1, step=1)


@register_test_case(module_factory=lambda: SliceScatterZeroDimModule())
def SliceScatterZeroDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6, 8), tu.rand(1, 8))


class SliceScatterNegativeEndModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, x, src):
        return torch.ops.aten.slice_scatter(x, src, dim=0, start=3, end=-1, step=1)


@register_test_case(module_factory=lambda: SliceScatterNegativeEndModule())
def SliceScatterNegativeEndModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6, 8), tu.rand(2, 8))


class SliceScatterNegativeDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, x, src):
        return torch.ops.aten.slice_scatter(x, src, dim=-2, start=0, end=1, step=1)


@register_test_case(module_factory=lambda: SliceScatterNegativeDimModule())
def SliceScatterNegativeDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6, 8), tu.rand(1, 8))


class SliceScatterStepVariationModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, x, src):
        return torch.ops.aten.slice_scatter(x, src, dim=1, start=0, end=1, step=2)


@register_test_case(module_factory=lambda: SliceScatterStepVariationModule())
def SliceScatterStepVariationModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6, 8), tu.rand(6, 1))


class SliceScatterStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([6, 8], torch.float32, True),
            ([6, 1], torch.float32, True),
        ]
    )
    def forward(self, x, src):
        return torch.ops.aten.slice_scatter(x, src, dim=1, start=0, end=1, step=1)


@register_test_case(module_factory=lambda: SliceScatterStaticModule())
def SliceScatterStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6, 8), tu.rand(6, 1))


class SelectScatterModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, x, src):
        return torch.ops.aten.select_scatter(x, src, dim=0, index=0)


@register_test_case(module_factory=lambda: SelectScatterModule())
def SelectScattertModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6, 8, 5), tu.rand(8, 5))


class SelectScatterStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([6, 8, 5], torch.float32, True),
            ([6, 5], torch.float32, True),
        ]
    )
    def forward(self, x, src):
        return torch.ops.aten.select_scatter(x, src, dim=1, index=0)


@register_test_case(module_factory=lambda: SelectScatterStaticModule())
def SelectScattertStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6, 8, 5), tu.rand(6, 5))


# ==============================================================================


class NarrowHorizontalTest(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.narrow(x, dim=0, start=0, length=2)


@register_test_case(module_factory=lambda: NarrowHorizontalTest())
def NarrowHorizontalTest_basic(module, tu: TestUtils):
    module.forward(tu.rand(6, 4, 3))


# ==============================================================================


class NarrowVerticalTest(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.narrow(x, dim=1, start=0, length=2)


@register_test_case(module_factory=lambda: NarrowVerticalTest())
def NarrowVerticalTest_basic(module, tu: TestUtils):
    module.forward(tu.rand(6, 4, 3))


# ==============================================================================


class NarrowHorizontalTest2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.narrow(x, dim=0, start=0, length=2)


@register_test_case(module_factory=lambda: NarrowHorizontalTest2())
def NarrowHorizontalTest2_basic(module, tu: TestUtils):
    module.forward(tu.rand(6, 4))


# ==============================================================================


class NarrowVerticalTest2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.narrow(x, dim=1, start=0, length=2)


@register_test_case(module_factory=lambda: NarrowVerticalTest2())
def NarrowVerticalTest2_basic(module, tu: TestUtils):
    module.forward(tu.rand(6, 4))


# ==============================================================================


class NarrowTensorHorizontalModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1, -1], torch.float32, True)])
    def forward(self, x):
        return torch.narrow(x, dim=1, start=torch.tensor(0), length=2)


@register_test_case(module_factory=lambda: NarrowTensorHorizontalModule())
def NarrowTensorHorizontalModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6, 4))


# ==============================================================================


class NarrowTensorVerticalModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1, -1], torch.float32, True)])
    def forward(self, x):
        return torch.narrow(x, dim=1, start=torch.tensor(1), length=2)


@register_test_case(module_factory=lambda: NarrowTensorVerticalModule())
def NarrowTensorVerticalModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6, 4))


# ==============================================================================


class SliceCopy_Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([10, 4, 4], torch.float32, True),
            ([4, 4, 4], torch.float32, True),
        ]
    )
    def forward(self, x, y):
        xslice = torch.ops.aten.slice(x, 0, 2, 6, 1)
        xslice.copy_(y)
        return x


@register_test_case(module_factory=lambda: SliceCopy_Module())
def SliceCopy_Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(10, 4, 4), tu.rand(4, 4, 4))


# ==============================================================================


class SliceCopyNegative_Module(torch.nn.Module):
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
    def forward(self, x, y):
        xslice = torch.ops.aten.slice(x, 0, 2, -4, 1)
        xslice.copy_(y)
        return x


@register_test_case(module_factory=lambda: SliceCopyNegative_Module())
def SliceCopyNegative_Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(10, 4, 4), tu.rand(4, 4, 4))


# ==============================================================================


class SliceCopyStartGreaterThanDimSize_Module(torch.nn.Module):
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
    def forward(self, x, y):
        xslice = torch.ops.aten.slice(x, 0, 100, 10, 1)
        xslice.copy_(y)
        return x


@register_test_case(module_factory=lambda: SliceCopyStartGreaterThanDimSize_Module())
def SliceCopyStartGreaterThanDimSize_Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(10, 4, 4), tu.rand(0, 4, 4))


# ==============================================================================


class SliceCopyEndGreaterThanDimSize_Module(torch.nn.Module):
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
    def forward(self, x, y):
        xslice = torch.ops.aten.slice(x, 0, 2, 100, 1)
        xslice.copy_(y)
        return x


@register_test_case(module_factory=lambda: SliceCopyEndGreaterThanDimSize_Module())
def SliceCopyEndGreaterThanDimSize_Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(10, 4, 4), tu.rand(8, 4, 4))


# ==============================================================================


class SliceCopyNonZeroDim_Module(torch.nn.Module):
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
    def forward(self, x, y):
        xslice = torch.ops.aten.slice(x, 1, 1, 3, 1)
        xslice.copy_(y)
        return x


@register_test_case(module_factory=lambda: SliceCopyNonZeroDim_Module())
def SliceCopyNonZeroDim_Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(10, 4, 4), tu.rand(10, 2, 4))


# ==============================================================================
class PrimListUnpackNumMismatchModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([5, 4, 3, 2, 1], torch.float32, True),
        ]
    )
    def forward(self, x):
        if len(x.shape) == 5:
            b0, t, c0, h0, w0 = x.shape
            b, c, h, w = torch.mul(b0, t), c0, h0, w0
        else:
            b1, c1, h1, w1 = x.shape
            b, c, h, w = b1, c1, h1, w1
        res = torch.reshape(x, [b, c, h, w])
        return res


@register_test_case(module_factory=lambda: PrimListUnpackNumMismatchModule())
def PrimListUnpackNumMismatchModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 4, 3, 2, 1))


# ==============================================================================


class UnbindIntListUnpack_Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 3, 4], torch.float32, True),
        ]
    )
    def forward(self, x):
        unbind_0, unbind_1 = torch.unbind(x, 0)
        return torch.ops.aten.sub(unbind_0, unbind_1)


@register_test_case(module_factory=lambda: UnbindIntListUnpack_Module())
def UnbindIntListUnpack_Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4))


# ==============================================================================


class UnbindIntGetItem_Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 3, 4], torch.float32, True),
        ]
    )
    def forward(self, x):
        unbind = torch.unbind(x, 0)
        return torch.ops.aten.sub(unbind[0], unbind[1])


@register_test_case(module_factory=lambda: UnbindIntGetItem_Module())
def UnbindIntGetItem_Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4))


# ==============================================================================


class SplitTensorGetItem_Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([3, 3, 4], torch.float32, True),
        ]
    )
    def forward(self, x):
        splits = torch.ops.aten.split(x, 2, 0)
        return torch.ops.aten.sub(splits[0], splits[1])


@register_test_case(module_factory=lambda: SplitTensorGetItem_Module())
def SplitTensorGetItem_Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 3, 4))


# ==============================================================================


class SplitTensorListUnpackModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([5, 3, 4], torch.float32, True),
        ]
    )
    def forward(self, x):
        x1, x2, x3 = torch.ops.aten.split(x, 2, 0)
        return x1 + x2 + x3


@register_test_case(module_factory=lambda: SplitTensorListUnpackModule())
def SplitTensorListUnpackModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 3, 4))


# ==============================================================================


class SplitTensorLastSmallerModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([8, 10, 12], torch.float32, True)])
    def forward(self, x):
        s0, s1, s2 = torch.ops.aten.split(x, 3, dim=0)
        return s2


@register_test_case(module_factory=lambda: SplitTensorLastSmallerModule())
def SplitTensorLastSmallerModule_basic(module, tu: TestUtils):
    # Splitting the first dimension with 8 elements into chunks of 3
    # will leave the last result to have 2 elements in that dimension.
    module.forward(tu.rand(8, 10, 12))


# ==============================================================================


class SplitTensorNegativeDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([10, 12, 6], torch.float32, True)])
    def forward(self, x):
        s0, s1, s2 = torch.ops.aten.split(x, 2, -1)
        return s1


@register_test_case(module_factory=lambda: SplitTensorNegativeDimModule())
def SplitTensorNegativeDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(10, 12, 6))


# ==============================================================================


class SplitWithSizesListUnpackModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([10, 12], torch.float32, True)])
    def forward(self, x):
        s0, s1, s2 = torch.ops.aten.split_with_sizes(x, [3, 4, 5], -1)
        return (s0, s1, s2)


@register_test_case(module_factory=lambda: SplitWithSizesListUnpackModule())
def SplitWithSizesListUnpackModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(10, 12))


# ==============================================================================


class ChunkListUnpack_Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 12, 2], torch.float32, True),
        ]
    )
    def forward(self, x):
        chunk_0, chunk_1, chunk_2 = torch.chunk(x, 3, 1)
        add = torch.ops.aten.add(chunk_0, chunk_1)
        sum = torch.ops.aten.add(add, chunk_2)
        return sum


@register_test_case(module_factory=lambda: ChunkListUnpack_Module())
def ChunkListUnpack_Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 12, 2))


# ==============================================================================


class ChunkListUnpackUneven_Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 13, 2], torch.float32, True),
        ]
    )
    def forward(self, x):
        a0, a1, a2, a3, a4 = torch.chunk(x, 6, 1)
        return a0, a1, a2, a3, a4


@register_test_case(module_factory=lambda: ChunkListUnpackUneven_Module())
def ChunkListUnpackUneven_Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 13, 2))


# ==============================================================================


class ChunkListUnpackDynamic_Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        chunk_0, chunk_1, chunk_2 = torch.chunk(x, 3, 1)
        add = torch.ops.aten.add(chunk_0, chunk_1)
        sum = torch.ops.aten.add(add, chunk_2)
        return sum


@register_test_case(module_factory=lambda: ChunkListUnpackDynamic_Module())
def ChunkListUnpackDynamic_Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 12, 2))


# ==============================================================================


class ChunkListUnpackUnevenDynamic_Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        chunk_0, chunk_1, chunk_2 = torch.chunk(x, 3, 1)
        return torch.ops.aten.add(chunk_0, chunk_1), chunk_2


@register_test_case(module_factory=lambda: ChunkListUnpackUnevenDynamic_Module())
def ChunkListUnpackUnevenDynamic_Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 13, 2))


# ==============================================================================


class SplitWithSizes_Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([5, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        split = torch.split(x, [2, 1, 2], dim=0)
        return split[0], split[1], split[2]


@register_test_case(module_factory=lambda: SplitWithSizes_Module())
def SplitWithSizes_Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 2, 2))


# ==============================================================================


class TensorSplitSections_GetItemModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 5], torch.float32, True),
        ]
    )
    def forward(self, x):
        split = torch.tensor_split(x, 3, dim=1)
        return split[0], split[1], split[2]


@register_test_case(module_factory=lambda: TensorSplitSections_GetItemModule())
def TensorSplitSections_GetItemModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 5))


class TensorSplitSections_ListUnpackModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 5], torch.float32, True),
        ]
    )
    def forward(self, x):
        a, b, c, d = torch.tensor_split(x, 4, dim=1)
        return a, b, c, d


@register_test_case(module_factory=lambda: TensorSplitSections_ListUnpackModule())
def TensorSplitSections_ListUnpackModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 5))


# ==============================================================================


class AsStridedWithOffsetModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 6, 60], torch.float32, True),
        ]
    )
    def forward(self, x):
        output_size = [6, 20]
        stride = [60, 1]
        slice = torch.ops.aten.slice.Tensor(x, 0, 1, 2)
        squeeze = torch.ops.aten.squeeze.dim(slice, 0)
        return torch.ops.aten.as_strided(
            squeeze, size=output_size, stride=stride, storage_offset=360
        )


@register_test_case(module_factory=lambda: AsStridedWithOffsetModule())
def AsStridedWithOffsetModule_basic(module, tu: TestUtils):
    module.forward(torch.rand(2, 6, 60))
