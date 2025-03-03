#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================


class ViewExpandModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([6, 4], torch.float32, True),
        ]
    )
    def forward(self, a):
        return a.view(2, 3, 4)


@register_test_case(module_factory=lambda: ViewExpandModule())
def ViewExpandModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6, 4))


# ==============================================================================


class ViewExpandOnesModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return a.view(1, 1, 1, 1, 1)


@register_test_case(module_factory=lambda: ViewExpandOnesModule())
def ViewExpandOnesModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1))


# ==============================================================================


class ViewExpandOnesBeforeAndAfterModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 1, 16, 1, 1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return a.view(1, 2, 1, 16, 1, 1, 1, 1)


@register_test_case(module_factory=lambda: ViewExpandOnesBeforeAndAfterModule())
def ViewExpandOnesBeforeAndAfterModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 1, 16, 1, 1))


# ==============================================================================


class ViewExpandOnesMiddleModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([3, 1, 2], torch.float32, True),
        ]
    )
    def forward(self, a):
        return a.view(3, 1, 1, 1, 1, 2)


@register_test_case(module_factory=lambda: ViewExpandOnesMiddleModule())
def ViewExpandOnesMiddleModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 1, 2))

    # ==============================================================================


class ViewCollapseOnesMiddleModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([3, 1, 1, 1, 1, 2], torch.float32, True),
        ]
    )
    def forward(self, a):
        return a.view(3, 1, 2)


@register_test_case(module_factory=lambda: ViewCollapseOnesMiddleModule())
def ViewCollapseOnesMiddleModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 1, 1, 1, 1, 2))


# ==============================================================================


class ViewDynamicExpandModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, 30, 384], torch.float32, True),
        ]
    )
    def forward(self, a):
        return a.view(2, 4, 5, 6, 12, 32)


@register_test_case(module_factory=lambda: ViewDynamicExpandModule())
def ViewDynamicExpandModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4, 30, 384))


# ==============================================================================


class SplitDimStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([12], torch.float32, True)])
    def forward(self, a):
        return torch.ops.prims.split_dim(a, 0, 4)


@register_test_case(module_factory=lambda: SplitDimStaticModule())
def SplitDimStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(12))


class SplitDimDynamicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1, -1], torch.float32, True)])
    def forward(self, a):
        return torch.ops.prims.split_dim(a, 0, 3)


@register_test_case(module_factory=lambda: SplitDimDynamicModule())
def SplitDimDynamicModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6, 5))


# ==============================================================================
#
class CollapseAllDimensionsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([2, 2, 2, 2], torch.float32, True)])
    def forward(self, a):
        return torch.ops.prims.collapse(a, 0, 3)


@register_test_case(module_factory=lambda: CollapseAllDimensionsModule())
def CollapseAllDimensionsModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 2, 2, 2))


# ==============================================================================
#
class CollapseRank1DynamicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1], torch.float32, True)])
    def forward(self, a):
        return torch.ops.prims.collapse(a, 0, 0)


@register_test_case(module_factory=lambda: CollapseRank1DynamicModule())
def CollapseRank1DynamicModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5))


# ==============================================================================
#
class CollapseStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([2, 3, 4], torch.float32, True)])
    def forward(self, a):
        return torch.ops.prims.collapse(a, 1, 2)


@register_test_case(module_factory=lambda: CollapseStaticModule())
def CollapseStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4))


# ==============================================================================
#
class CollapsePartialDynamicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1, -1, 4, 5], torch.float32, True)])
    def forward(self, a):
        return torch.ops.prims.collapse(a, 1, 2)


@register_test_case(module_factory=lambda: CollapsePartialDynamicModule())
def CollapsePartialDynamicModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4, 5))


class CollapseFullDynamicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1, -1, -1], torch.float32, True)])
    def forward(self, a):
        return torch.ops.prims.collapse(a, 0, 1)


@register_test_case(module_factory=lambda: CollapseFullDynamicModule())
def CollapseFullDynamicModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 5))


# ==============================================================================


class ViewDynamicExpandWithAtenSizeIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return a.view(a.size(0), a.size(1), 12, 32)


@register_test_case(module_factory=lambda: ViewDynamicExpandWithAtenSizeIntModule())
def ViewDynamicExpandWithAtenSizeIntModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4, 384))


# ==============================================================================


class ViewCollapseModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return a.view(8)


@register_test_case(module_factory=lambda: ViewCollapseModule())
def ViewCollapseModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4))


# ==============================================================================


class ViewCollapseDynamicWithAtenSizeIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1, -1, -1], torch.float32, True),
            ([], torch.int64, True),
            ([], torch.int64, True),
        ]
    )
    def forward(self, a, b, c):
        return a.view(a.size(0), int(b), int(c), a.size(3), 384)


@register_test_case(module_factory=lambda: ViewCollapseDynamicWithAtenSizeIntModule())
def ViewCollapseDynamicWithAtenSizeIntModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 5, 4, 12, 32), torch.tensor(3), torch.tensor(5))


# ==============================================================================


class ViewExpandCollapseWithOnesModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 4, 8, 8], torch.float32, True),
        ]
    )
    def forward(self, a):
        return a.view(2, 1, 1, 4, 64)


@register_test_case(module_factory=lambda: ViewExpandCollapseWithOnesModule())
def ViewExpandCollapseWithOnesModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4, 8, 8))


# ==============================================================================


class ViewExpandCollapseModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 4, 8, 16, 4], torch.float32, True),
        ]
    )
    def forward(self, a):
        return a.view(8, 2, 4, 16, 2, 2)


@register_test_case(module_factory=lambda: ViewExpandCollapseModule())
def ViewExpandCollapseModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4, 8, 16, 4))


# ==============================================================================


class ViewDynamicExpandCollapseModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, 4, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return a.view(2, 1, 4, 64)


@register_test_case(module_factory=lambda: ViewDynamicExpandCollapseModule())
def ViewDynamicExpandCollapseModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4, 8, 8))


# ==============================================================================


class ViewDynamicExpandCollapseWithParallelUnknownDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 3, -1, 5], torch.float32, True),
        ]
    )
    def forward(self, a):
        return a.view(2, -1, 6)


@register_test_case(
    module_factory=lambda: ViewDynamicExpandCollapseWithParallelUnknownDimModule()
)
def ViewDynamicExpandCollapseWithParallelUnknownDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4, 5))


# ==============================================================================


class ViewDynamicExpandCollapseWithAtenIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return a.view(2, 1, a.size(1), 64)


@register_test_case(module_factory=lambda: ViewDynamicExpandCollapseWithAtenIntModule())
def ViewDynamicExpandCollapseWithAtenIntModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4, 8, 8))


# ==============================================================================


class ViewTwoToThreeStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([3, 2], torch.float32, True),
        ]
    )
    def forward(self, a):
        return a.view(2, 3)


@register_test_case(module_factory=lambda: ViewTwoToThreeStaticModule())
def ViewTwoToThreeStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 2))


# ==============================================================================


class ViewTwoFiveThreeStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([3, 5, 2], torch.float32, True),
        ]
    )
    def forward(self, a):
        return a.view(2, 5, 3)


@register_test_case(module_factory=lambda: ViewTwoFiveThreeStaticModule())
def ViewTwoFiveThreeStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5, 2))


# ==============================================================================


class ViewOffsetTestStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 3, 2, 2, 5, 6], torch.float32, True),
        ]
    )
    def forward(self, a):
        return a.view(2, 3, 4, 6, 5)


@register_test_case(module_factory=lambda: ViewOffsetTestStaticModule())
def ViewOffsetTestStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 2, 2, 5, 6))


# ==============================================================================


class ViewOffsetBackwardTestStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 3, 4, 5, 6], torch.float32, True),
        ]
    )
    def forward(self, a):
        return a.view(2, 3, 2, 2, 6, 5)


@register_test_case(module_factory=lambda: ViewOffsetBackwardTestStaticModule())
def ViewOffsetBackwardTestStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4, 5, 6))


# ==============================================================================


class View1DFoldModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return a.view(-1)


@register_test_case(module_factory=lambda: View1DFoldModule())
def View1DFoldModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(32))


# ==============================================================================


class ViewCollapseInferredDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 3, 4], torch.float32, True),
        ]
    )
    def forward(self, a):
        return a.view(-1, 4)


@register_test_case(module_factory=lambda: ViewCollapseInferredDimModule())
def ViewCollapseInferredDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4))


# ==============================================================================


class ViewExpandInferredDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 6], torch.float32, True),
        ]
    )
    def forward(self, a):
        return a.view(3, -1, 2)


@register_test_case(module_factory=lambda: ViewExpandInferredDimModule())
def ViewExpandInferredDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 6))


# ==============================================================================


class ViewExpandDynamicDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1, -1, 128], torch.float32, True),
        ]
    )
    def forward(self, a):
        return a.view(16, 1, 128)


@register_test_case(module_factory=lambda: ViewExpandDynamicDimModule())
def ViewExpandDynamicDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 16, 128))


# ==============================================================================


class ViewFlattenAndExpandModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return a.view(a.size(0), a.size(1))


@register_test_case(module_factory=lambda: ViewFlattenAndExpandModule())
def ViewFlattenAndExpandModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(64, 128))


# ==============================================================================


class ViewSizeFromOtherTensor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1, -1], torch.float32, True),
            ([1, -1, 10], torch.float32, True),
        ]
    )
    def forward(self, x, y):
        return torch.ops.aten.view(y, (torch.ops.aten.size(x, 1), 10))


@register_test_case(module_factory=lambda: ViewSizeFromOtherTensor())
def ViewSizeFromOtherTensor_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 7), tu.rand(1, 7, 10))


# ==============================================================================


class UnsafeViewExpandModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([6, 4], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten._unsafe_view(a, [2, 3, 4])


@register_test_case(module_factory=lambda: UnsafeViewExpandModule())
def UnsafeViewExpandModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6, 4))


# ==============================================================================


class UnsafeViewDynamicExpandModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, 30, 384], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten._unsafe_view(a, [2, 4, 5, 6, 12, 32])


@register_test_case(module_factory=lambda: UnsafeViewDynamicExpandModule())
def UnsafeViewDynamicExpandModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4, 30, 384))


# ==============================================================================


class UnsafeViewDynamicExpandWithAtenSizeIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten._unsafe_view(a, [a.size(0), a.size(1), 12, 32])


@register_test_case(
    module_factory=lambda: UnsafeViewDynamicExpandWithAtenSizeIntModule()
)
def UnsafeViewDynamicExpandWithAtenSizeIntModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4, 384))


# ==============================================================================


class UnsafeViewCollapseModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten._unsafe_view(a, [8])


@register_test_case(module_factory=lambda: UnsafeViewCollapseModule())
def UnsafeViewCollapseModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4))


# ==============================================================================


class UnsafeViewCollapseDynamicWithAtenSizeIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1, -1, -1], torch.float32, True),
            ([], torch.int64, True),
            ([], torch.int64, True),
        ]
    )
    def forward(self, a, b, c):
        return torch.ops.aten._unsafe_view(
            a, [a.size(0), int(b), int(c), a.size(3), 384]
        )


@register_test_case(
    module_factory=lambda: UnsafeViewCollapseDynamicWithAtenSizeIntModule()
)
def UnsafeViewCollapseDynamicWithAtenSizeIntModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 5, 4, 12, 32), torch.tensor(3), torch.tensor(5))


# ==============================================================================


class UnsafeView1DFoldModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten._unsafe_view(a, [-1])


@register_test_case(module_factory=lambda: UnsafeView1DFoldModule())
def UnsafeView1DFoldModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(32))


# ==============================================================================


class ReshapeAsModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([4, 3], torch.float32, True),
            ([2, 6], torch.float32, True),
        ]
    )
    def forward(self, a, b):
        return torch.ops.aten.reshape_as(a, b)


@register_test_case(module_factory=lambda: ReshapeAsModule())
def ReshapeAsModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 3), tu.rand(2, 6))


# ==============================================================================


class ReshapeExpandModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return a.reshape(12, 32)


@register_test_case(module_factory=lambda: ReshapeExpandModule())
def ReshapeExpandModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(384))


# ==============================================================================


class ReshapeCollapseModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.reshape(a, (-1,))


@register_test_case(module_factory=lambda: ReshapeCollapseModule())
def ReshapeCollapseModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4))


# ==============================================================================


class ViewNoChange1dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return a.view(6)


@register_test_case(module_factory=lambda: ViewNoChange1dModule())
def ViewNoChange1dModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6))


class ViewNoChange2dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return a.view(5, 6)


@register_test_case(module_factory=lambda: ViewNoChange2dModule())
def ViewNoChange2dModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 6))


class ViewNoChange3dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return a.view(4, 5, 6)


@register_test_case(module_factory=lambda: ViewNoChange3dModule())
def ViewNoChange3dModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 5, 6))


class ViewNoChangeStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([4, 5, 6], torch.float32, True),
        ]
    )
    def forward(self, a):
        return a.view(4, 5, 6)


@register_test_case(module_factory=lambda: ViewNoChangeStaticModule())
def ViewNoChangeStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 5, 6))


class ViewNegativeStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1, 128], torch.float32, True),
        ]
    )
    def forward(self, a):
        return a.view(-1, 128)


@register_test_case(module_factory=lambda: ViewNegativeStaticModule())
def ViewNegativeStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 128))


class ViewSizeDimFollowedByExpandedOnesModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return a.view(a.size(0), 1, 1, 1)


@register_test_case(module_factory=lambda: ViewSizeDimFollowedByExpandedOnesModule())
def ViewSizeDimFollowedByExpandedOnesModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(128))


class ViewSizeDimFollowedByCollapsedOnesModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, 1, 1, 1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return a.view(a.size(0))


@register_test_case(module_factory=lambda: ViewSizeDimFollowedByCollapsedOnesModule())
def ViewSizeDimFollowedByCollapsedOnesModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(128, 1, 1, 1))


class ViewSizeDimLedByExpandedOnesModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return a.view(1, 1, 1, a.size(0))


@register_test_case(module_factory=lambda: ViewSizeDimLedByExpandedOnesModule())
def ViewSizeDimLedByExpandedOnesModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(128))


class ViewSizeDimLedByCollapsedOnesModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1, 1, 1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return a.view(a.size(3))


@register_test_case(module_factory=lambda: ViewSizeDimLedByCollapsedOnesModule())
def ViewSizeDimLedByCollapsedOnesModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 1, 128))


class ViewSizeDimLedAndFollowedByExpandedOnesModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return a.view(1, 1, 1, a.size(0), 1, 1, 1)


@register_test_case(
    module_factory=lambda: ViewSizeDimLedAndFollowedByExpandedOnesModule()
)
def ViewSizeDimLedAndFollowedByExpandedOnesModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(128))


class ViewSizeDimLedAndFollowedByCollapsedOnesModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1, 1, 1, -1, 1, 1, 1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return a.view(a.size(3))


@register_test_case(
    module_factory=lambda: ViewSizeDimLedAndFollowedByCollapsedOnesModule()
)
def ViewSizeDimLedAndFollowedByCollapsedOnesModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 1, 128, 1, 1, 1))


# ==============================================================================


class ReshapeAliasExpandModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.reshape_alias = torch.ops.aten._reshape_alias

    @export
    @annotate_args(
        [
            None,
            ([-1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten._reshape_alias(a, size=(12, 32), stride=(32, 1))


@register_test_case(module_factory=lambda: ReshapeAliasExpandModule())
def ReshapeAliasExpandModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(384))


# ==============================================================================


class ReshapeDynamicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return a.view(a.size(1), a.size(0))


@register_test_case(module_factory=lambda: ReshapeDynamicModule())
def ReshapeDynamicModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


# ==============================================================================


class ViewDtypeStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([12, 1], torch.float32, True),
        ]
    )
    def forward(self, a):
        res = a.view(torch.int8)
        return res


@register_test_case(module_factory=lambda: ViewDtypeStaticModule())
def ViewDtypeStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(12, 1))


# ==============================================================================


class ReshapeAliasCollapseModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten._reshape_alias(a, (8,), (1,))


@register_test_case(module_factory=lambda: ReshapeAliasCollapseModule())
def ReshapeAliasCollapseModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4))


# ==============================================================================


class UnflattenIntStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([3, 24, 5], torch.float32, True),
        ]
    )
    def forward(self, inputs):
        return torch.ops.aten.unflatten(inputs, 1, [2, 4, 3])


@register_test_case(module_factory=lambda: UnflattenIntStaticModule())
def UnflattenIntStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 24, 5))


class UnflattenIntNegativeOneDimStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([5, 12, 3], torch.float32, True),
        ]
    )
    def forward(self, inputs):
        return torch.ops.aten.unflatten(inputs, -2, [2, 2, 3, 1, 1])


@register_test_case(module_factory=lambda: UnflattenIntNegativeOneDimStaticModule())
def UnflattenIntNegativeOneDimStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 12, 3))


class UnflattenIntNegativeOneSizeStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([5, 12, 3], torch.float32, True),
        ]
    )
    def forward(self, inputs):
        return torch.ops.aten.unflatten(inputs, -2, [2, -1, 3, 1, 1])


@register_test_case(module_factory=lambda: UnflattenIntNegativeOneSizeStaticModule())
def UnflattenIntNegativeOneSizeStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 12, 3))


# ==============================================================================


class EinsumStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([3, 2, 4], torch.float32, True),
            ([5, 4, 6], torch.float32, True),
            ([3, 7, 6], torch.float32, True),
        ]
    )
    def forward(self, tensor1, tensor2, tensor3):
        return torch.ops.aten.einsum("bqe,ked,btd->bqtk", [tensor1, tensor2, tensor3])


@register_test_case(module_factory=lambda: EinsumStaticModule())
def EinsumStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 2, 4), tu.rand(5, 4, 6), tu.rand(3, 7, 6))


class EinsumStaticFourDimensionModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([3, 4, 5, 6], torch.float32, True),
            ([3, 7, 5, 6], torch.float32, True),
        ]
    )
    def forward(self, tensor1, tensor2):
        return torch.ops.aten.einsum("blhd,bshd->blhs", [tensor1, tensor2])


@register_test_case(module_factory=lambda: EinsumStaticFourDimensionModule())
def EinsumStaticFourDimensionModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5, 6), tu.rand(3, 7, 5, 6))


class EinsumStaticDiagonalDimensionModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([5, 5, 4, 4], torch.float32, True),
            ([5, 4, 5, 4], torch.float32, True),
        ]
    )
    def forward(self, tensor1, tensor2):
        return torch.ops.aten.einsum("iijj,ijij->ji", [tensor1, tensor2])


@register_test_case(module_factory=lambda: EinsumStaticDiagonalDimensionModule())
def EinsumStaticDiagonalDimensionModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 5, 4, 4), tu.rand(5, 4, 5, 4))


class EinsumStaticContractRhsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([3, 4, 5], torch.float32, True),
            ([4, 5], torch.float32, True),
        ]
    )
    def forward(self, tensor1, tensor2):
        return torch.ops.aten.einsum("abc,bc->a", [tensor1, tensor2])


@register_test_case(module_factory=lambda: EinsumStaticContractRhsModule())
def EinsumStaticContractRhsModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5), tu.rand(4, 5))


class EinsumStaticWithEllipsisSlicingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([3, 4, 6], torch.float32, True),
            ([3, 6, 5], torch.float32, True),
        ]
    )
    def forward(self, tensor1, tensor2):
        return torch.ops.aten.einsum("...mn,...nd->...md", [tensor1, tensor2])


@register_test_case(module_factory=lambda: EinsumStaticWithEllipsisSlicingModule())
def EinsumStaticWithEllipsisSlicingModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 6), tu.rand(3, 6, 5))


class EinsumStaticWithEllipsisSlicingAndBroadcastModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 6, 4, 5], torch.float32, True),
            ([6, 5], torch.float32, True),
        ]
    )
    def forward(self, tensor1, tensor2):
        # should be abnd,bd -> abn
        return torch.ops.aten.einsum("...nd,...d->...n", [tensor1, tensor2])


@register_test_case(
    module_factory=lambda: EinsumStaticWithEllipsisSlicingAndBroadcastModule()
)
def EinsumStaticWithEllipsisSlicingAndBroadcastModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 6, 4, 5), tu.rand(6, 5))


class InterpolateModule(torch.nn.Module):
    def __init__(
        self,
        size=None,
        scale_factor=None,
        mode="nearest",
        align_corners=None,
        recompute_scale_factor=None,
        antialias=False,
    ):
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor
        self.antialias = antialias
        super().__init__()

    def _forward(self, input):
        return torch.nn.functional.interpolate(
            input,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            recompute_scale_factor=self.recompute_scale_factor,
            antialias=self.antialias,
        )


class InterpolateStaticModule(InterpolateModule):
    @export
    @annotate_args(
        [
            None,
            ([1, 1, 4, 5], torch.float32, True),
        ]
    )
    def forward(self, input):
        return self._forward(input)


class InterpolateDynamicModule(InterpolateModule):
    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, input):
        return self._forward(input)


@register_test_case(
    module_factory=lambda: InterpolateStaticModule(
        scale_factor=0.41, mode="bilinear", align_corners=True
    )
)
def InterpolateStaticModule_scales_bilinear_align_corners(module, tu: TestUtils):
    input = torch.arange(20).to(dtype=torch.float32)
    input = input.reshape((1, 1, 4, 5))
    module.forward(input)


@register_test_case(
    module_factory=lambda: InterpolateDynamicModule(size=(2, 7), mode="nearest")
)
def InterpolateDynamicModule_sizes_nearest(module, tu: TestUtils):
    input = torch.arange(20).to(dtype=torch.float32)
    input = input.reshape((1, 1, 4, 5))
    module.forward(input)


@register_test_case(
    module_factory=lambda: InterpolateDynamicModule(size=(2, 7), mode="bilinear")
)
def InterpolateDynamicModule_sizes_bilinear(module, tu: TestUtils):
    input = torch.arange(20).to(dtype=torch.float32)
    input = input.reshape((1, 1, 4, 5))
    module.forward(input)


@register_test_case(
    module_factory=lambda: InterpolateDynamicModule(
        scale_factor=(1.9, 2.4), mode="bilinear", recompute_scale_factor=True
    )
)
def InterpolateDynamicModule_scales_recompute_bilinear(module, tu: TestUtils):
    input = torch.arange(20).to(dtype=torch.float32)
    input = input.reshape((1, 1, 4, 5))
    module.forward(input)


# ==============================================================================


class Atleast1dModule0dInput(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.atleast_1d(x)


@register_test_case(module_factory=lambda: Atleast1dModule0dInput())
def Atleast1dModule0dInput_basic(module, tu: TestUtils):
    module.forward(tu.rand())


class Atleast1dModule1dInput(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([4], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.atleast_1d(x)


@register_test_case(module_factory=lambda: Atleast1dModule1dInput())
def Atleast1dModule1dInput_basic(module, tu: TestUtils):
    module.forward(tu.rand(4))


class Atleast2dModule0dInput(torch.nn.Module):
    @export
    @annotate_args([None, [(), torch.float32, True]])
    def forward(self, x):
        return torch.ops.aten.atleast_2d(x)


@register_test_case(module_factory=lambda: Atleast2dModule0dInput())
def Atleast2dModule0dInput_basic(module, tu: TestUtils):
    module.forward(tu.rand())


class Atleast2dModule1dInput(torch.nn.Module):
    @export
    @annotate_args([None, [(10,), torch.float32, True]])
    def forward(self, x):
        return torch.ops.aten.atleast_2d(x)


@register_test_case(module_factory=lambda: Atleast2dModule1dInput())
def Atleast2dModule1dInput_basic(module, tu: TestUtils):
    module.forward(tu.rand(10))


class Atleast2dModule2dInput(torch.nn.Module):
    @export
    @annotate_args([None, [(3, 4), torch.float32, True]])
    def forward(self, x):
        return torch.ops.aten.atleast_2d(x)


@register_test_case(module_factory=lambda: Atleast2dModule2dInput())
def Atleast2dModule2dInput_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


class Atleast2dModule3dInput(torch.nn.Module):
    @export
    @annotate_args([None, [(2, 3, 4), torch.float32, True]])
    def forward(self, x):
        return torch.ops.aten.atleast_2d(x)


@register_test_case(module_factory=lambda: Atleast2dModule3dInput())
def Atleast2dModule3dInput_basic(module, tu: TestUtils):
    result = module.forward(tu.rand(2, 3, 4))


class Atleast3dModule0dInput(torch.nn.Module):
    @export
    @annotate_args([None, [(), torch.float32, True]])
    def forward(self, x):
        return torch.ops.aten.atleast_3d(x)


@register_test_case(module_factory=lambda: Atleast3dModule0dInput())
def Atleast3dModule0dInput_basic(module, tu: TestUtils):
    result = module.forward(tu.rand())


class Atleast3dModule1dInput(torch.nn.Module):
    @export
    @annotate_args([None, [(10,), torch.float32, True]])
    def forward(self, x):
        return torch.ops.aten.atleast_3d(x)


@register_test_case(module_factory=lambda: Atleast3dModule1dInput())
def Atleast3dModule1dInput_basic(module, tu: TestUtils):
    result = module.forward(tu.rand(10))


class Atleast3dModule2dInput(torch.nn.Module):
    @export
    @annotate_args([None, [(4, 5), torch.float32, True]])
    def forward(self, x):
        return torch.ops.aten.atleast_3d(x)


@register_test_case(module_factory=lambda: Atleast3dModule2dInput())
def Atleast3dModule2dInput_basic(module, tu: TestUtils):
    result = module.forward(tu.rand(4, 5))


class Atleast3dModule3dInput(torch.nn.Module):
    @export
    @annotate_args([None, [(2, 3, 4), torch.float32, True]])
    def forward(self, x):
        return torch.ops.aten.atleast_3d(x)


@register_test_case(module_factory=lambda: Atleast3dModule3dInput())
def Atleast3dModule3dInput_basic(module, tu: TestUtils):
    result = module.forward(tu.rand(2, 3, 4))


# ==============================================================================


class Rot90BasicModule(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([4, 5], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.rot90(
            a,
            k=1,
            dims=(
                0,
                1,
            ),
        )


@register_test_case(module_factory=lambda: Rot90BasicModule())
def Rot90BasicModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 5))


class Rot90DynamicDimsModule(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.rot90(
            a,
            k=1,
            dims=(
                0,
                1,
            ),
        )


@register_test_case(module_factory=lambda: Rot90DynamicDimsModule())
def Rot90DynamicDimsModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6, 2, 4))


class Rot90MultipleRotationsModule(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([7, 4, 6], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.rot90(
            a,
            k=6,
            dims=(
                1,
                2,
            ),
        )


@register_test_case(module_factory=lambda: Rot90MultipleRotationsModule())
def Rot90MultipleRotationsModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(7, 4, 6))


class Rot90NegativeOddRotationsModule(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([7, 4, 6, 5, 3], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.rot90(
            a,
            k=-5,
            dims=(
                1,
                2,
            ),
        )


@register_test_case(module_factory=lambda: Rot90NegativeOddRotationsModule())
def Rot90NegativeOddRotationsModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(7, 4, 6, 5, 3))


class Rot90NegativeEvenRotationsModule(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([6, 5, 1, 7, 3], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.rot90(
            a,
            k=-6,
            dims=(
                1,
                -2,
            ),
        )


@register_test_case(module_factory=lambda: Rot90NegativeEvenRotationsModule())
def Rot90NegativeEvenRotationsModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6, 5, 1, 7, 3))


# ==============================================================================


class Unfold_Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([6, 4], torch.float32, True),
        ]
    )
    def forward(self, x):
        return x.unfold(0, 2, 2)


@register_test_case(module_factory=lambda: Unfold_Module())
def Unfold_Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(6, 4))


class Unfold_Module_Negative_Dim(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([6, 4, 4, 4], torch.float32, True),
        ]
    )
    def forward(self, x):
        return x.unfold(-1, 2, 1)


@register_test_case(module_factory=lambda: Unfold_Module_Negative_Dim())
def Unfold_Module_Rank_4(module, tu: TestUtils):
    module.forward(tu.rand(6, 4, 4, 4))


class Unfold_Module_Rank_Zero(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([], torch.float32, True),
        ]
    )
    def forward(self, x):
        return x.unfold(0, 1, 1)


@register_test_case(module_factory=lambda: Unfold_Module_Rank_Zero())
def Unfold_Module_Rank_Zero_basic(module, tu: TestUtils):
    module.forward(tu.rand())


class Unfold_Module_Rank_Zero_Size_Zero(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([], torch.float32, True),
        ]
    )
    def forward(self, x):
        return x.unfold(0, 0, 1)


@register_test_case(module_factory=lambda: Unfold_Module_Rank_Zero_Size_Zero())
def Unfold_Module_Rank_Zero_Size_Zero_basic(module, tu: TestUtils):
    module.forward(tu.rand())


class Unfold_Module_Dynamic(torch.nn.Module):
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
        return x.unfold(1, 2, 1)


@register_test_case(module_factory=lambda: Unfold_Module_Dynamic())
def Unfold_Module_Dynamic_basic(module, tu: TestUtils):
    module.forward(tu.rand(6, 4, 4, 4))


# ==============================================================================


class Aten_TrilinearModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([3, 3, 3], torch.float32, True),
            ([3, 3, 3], torch.float32, True),
            ([3, 3, 3], torch.float32, True),
        ]
    )
    def forward(self, i1, i2, i3):
        return torch.ops.aten._trilinear(
            i1, i2, i3, expand1=[], expand2=[], expand3=[], sumdim=[], unroll_dim=0
        )


@register_test_case(module_factory=lambda: Aten_TrilinearModule())
def Aten_TrilinearModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 3, 3), tu.rand(3, 3, 3), tu.rand(3, 3, 3))


class Aten_TrilinearModuleSumdims(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 6], torch.float32, True),
            ([2, 3, 6], torch.float32, True),
            ([2, 3, 6], torch.float32, True),
        ]
    )
    def forward(self, i1, i2, i3):
        return torch.ops.aten._trilinear(
            i1, i2, i3, expand1=[1], expand2=[], expand3=[], sumdim=[0, 2], unroll_dim=0
        )


@register_test_case(module_factory=lambda: Aten_TrilinearModuleSumdims())
def Aten_TrilinearModuleSumdims_basic(module, tu: TestUtils):
    return module.forward(tu.rand(2, 6), tu.rand(2, 3, 6), tu.rand(2, 3, 6))


class Aten_TrilinearModuleSumAllDims(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 6], torch.float32, True),
            ([2, 3, 6], torch.float32, True),
            ([2, 3, 6], torch.float32, True),
        ]
    )
    def forward(self, i1, i2, i3):
        return torch.ops.aten._trilinear(
            i1,
            i2,
            i3,
            expand1=[1],
            expand2=[],
            expand3=[],
            sumdim=[0, 1, 2],
            unroll_dim=0,
        )


@register_test_case(module_factory=lambda: Aten_TrilinearModuleSumAllDims())
def Aten_TrilinearModuleSumAllDims_basic(module, tu: TestUtils):
    return module.forward(tu.rand(2, 6), tu.rand(2, 3, 6), tu.rand(2, 3, 6))


class Aten_TrilinearModuleVaryingRanks(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 6], torch.float32, True),
            ([2, 3, 6], torch.float32, True),
            ([6], torch.float32, True),
        ]
    )
    def forward(self, i1, i2, i3):
        return torch.ops.aten._trilinear(
            i1,
            i2,
            i3,
            expand1=[1],
            expand2=[],
            expand3=[0, 1],
            sumdim=[0],
            unroll_dim=0,
        )


@register_test_case(module_factory=lambda: Aten_TrilinearModuleVaryingRanks())
def Aten_TrilinearModuleVaryingRanks_basic(module, tu: TestUtils):
    return module.forward(tu.rand(2, 6), tu.rand(2, 3, 6), tu.rand(6))


class Aten_TrilinearModuleVaryingRanksUnorderedExpands(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 6], torch.float32, True),
            ([2, 3, 6], torch.float32, True),
            ([6], torch.float32, True),
        ]
    )
    def forward(self, i1, i2, i3):
        return torch.ops.aten._trilinear(
            i1,
            i2,
            i3,
            expand1=[1],
            expand2=[],
            expand3=[1, 0],
            sumdim=[2, 0],
            unroll_dim=0,
        )


@register_test_case(
    module_factory=lambda: Aten_TrilinearModuleVaryingRanksUnorderedExpands()
)
def Aten_TrilinearModuleVaryingRanksUnorderedExpands_basic(module, tu: TestUtils):
    return module.forward(tu.rand(2, 6), tu.rand(2, 3, 6), tu.rand(6))


class Aten_TrilinearModuleZerodDimBug(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 3, 6], torch.float32, True),
            ([2, 3, 6], torch.float32, True),
            ([2, 3, 6], torch.float32, True),
        ]
    )
    def forward(self, i1, i2, i3):
        return torch.ops.aten._trilinear(
            i1, i2, i3, expand1=[0], expand2=[0], expand3=[0], sumdim=[2], unroll_dim=0
        )


@register_test_case(module_factory=lambda: Aten_TrilinearModuleZerodDimBug())
def Aten_TrilinearModuleZerodDimBug_basic(module, tu: TestUtils):
    return module.forward(tu.rand(2, 3, 6), tu.rand(2, 3, 6), tu.rand(2, 3, 6))
