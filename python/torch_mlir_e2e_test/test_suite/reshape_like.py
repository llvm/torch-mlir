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
    @annotate_args([
        None,
        ([6, 4], torch.float32, True),
    ])

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
    @annotate_args([
        None,
        ([1], torch.float32, True),
    ])

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
    @annotate_args([
        None,
        ([2, 1, 16, 1, 1], torch.float32, True),
    ])

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
    @annotate_args([
        None,
        ([3, 1, 2], torch.float32, True),
    ])

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
    @annotate_args([
        None,
        ([3, 1, 1, 1, 1, 2], torch.float32, True),
    ])

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
    @annotate_args([
        None,
        ([-1, -1, 30, 384], torch.float32, True),
    ])

    def forward(self, a):
        return a.view(2, 4, 5, 6, 12, 32)

@register_test_case(module_factory=lambda: ViewDynamicExpandModule())
def ViewDynamicExpandModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4, 30, 384))

# ==============================================================================

class ViewDynamicExpandWithAtenSizeIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])

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
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])

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
    @annotate_args([
        None,
        ([-1, -1, -1, -1, -1, -1], torch.float32, True),
        ([], torch.int64, True),
        ([], torch.int64, True),
    ])

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
    @annotate_args([
        None,
        ([2, 4, 8, 8], torch.float32, True),
    ])

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
    @annotate_args([
        None,
        ([2, 4, 8, 16, 4], torch.float32, True),
    ])

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
    @annotate_args([
        None,
        ([-1, 4, -1, -1], torch.float32, True),
    ])

    def forward(self, a):
        return a.view(2, 1, 4, 64)

@register_test_case(module_factory=lambda: ViewDynamicExpandCollapseModule())
def ViewDynamicExpandCollapseModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4, 8, 8))

# ==============================================================================

class ViewDynamicExpandCollapseWithAtenIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ])

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
    @annotate_args([
        None,
        ([3, 2], torch.float32, True),
    ])

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
    @annotate_args([
        None,
        ([3, 5, 2], torch.float32, True),
    ])

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
    @annotate_args([
        None,
        ([2, 3, 2, 2, 5, 6], torch.float32, True),
    ])

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
    @annotate_args([
        None,
        ([2, 3, 4, 5, 6], torch.float32, True),
    ])

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
    @annotate_args([
        None,
        ([-1], torch.float32, True),
    ])

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
    @annotate_args([
        None,
        ([2, 3, 4], torch.float32, True),
    ])

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
    @annotate_args([
        None,
        ([2, 6], torch.float32, True),
    ])

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
    @annotate_args([
        None,
        ([1, -1, 128], torch.float32, True),
    ])

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
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])

    def forward(self, a):
        return a.view(a.size(0), a.size(1))

@register_test_case(module_factory=lambda: ViewFlattenAndExpandModule())
def ViewFlattenAndExpandModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(64,128))

# ==============================================================================

class UnsafeViewExpandModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([6, 4], torch.float32, True),
    ])

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
    @annotate_args([
        None,
        ([-1, -1, 30, 384], torch.float32, True),
    ])

    def forward(self, a):
        return torch.ops.aten._unsafe_view(a,[2, 4, 5, 6, 12, 32])

@register_test_case(module_factory=lambda: UnsafeViewDynamicExpandModule())
def UnsafeViewDynamicExpandModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4, 30, 384))

# ==============================================================================

class UnsafeViewDynamicExpandWithAtenSizeIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])

    def forward(self, a):
        return torch.ops.aten._unsafe_view(a, [a.size(0), a.size(1), 12, 32])

@register_test_case(module_factory=lambda: UnsafeViewDynamicExpandWithAtenSizeIntModule())
def UnsafeViewDynamicExpandWithAtenSizeIntModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4, 384))

# ==============================================================================

class UnsafeViewCollapseModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])

    def forward(self, a):
        return torch.ops.aten._unsafe_view(a,[8])

@register_test_case(module_factory=lambda: UnsafeViewCollapseModule())
def UnsafeViewCollapseModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4))

# ==============================================================================

class UnsafeViewCollapseDynamicWithAtenSizeIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1, -1, -1], torch.float32, True),
        ([], torch.int64, True),
        ([], torch.int64, True),
    ])

    def forward(self, a, b, c):
        return torch.ops.aten._unsafe_view(a, [a.size(0), int(b), int(c), a.size(3), 384])

@register_test_case(module_factory=lambda: UnsafeViewCollapseDynamicWithAtenSizeIntModule())
def UnsafeViewCollapseDynamicWithAtenSizeIntModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 5, 4, 12, 32), torch.tensor(3), torch.tensor(5))

# ==============================================================================

class UnsafeView1DFoldModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.float32, True),
    ])

    def forward(self, a):
        return torch.ops.aten._unsafe_view(a, [-1])

@register_test_case(module_factory=lambda: UnsafeView1DFoldModule())
def UnsafeView1DFoldModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(32))

# ==============================================================================

class ReshapeExpandModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.float32, True),
    ])

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
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])

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
    @annotate_args([
        None,
        ([-1], torch.float32, True),
    ])

    def forward(self, a):
        return a.view(6)

@register_test_case(module_factory=lambda: ViewNoChange1dModule())
def ViewNoChange1dModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6))


class ViewNoChange2dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])

    def forward(self, a):
        return a.view(5, 6)

@register_test_case(module_factory=lambda: ViewNoChange2dModule())
def ViewNoChange2dModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 6))


class ViewNoChange3dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])

    def forward(self, a):
        return a.view(4, 5, 6)

@register_test_case(module_factory=lambda: ViewNoChange3dModule())
def ViewNoChange3dModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 5, 6))


class ViewNoChangeStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([4, 5, 6], torch.float32, True),
    ])

    def forward(self, a):
        return a.view(4, 5, 6)

@register_test_case(module_factory=lambda: ViewNoChangeStaticModule())
def ViewNoChangeStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 5, 6))

# ==============================================================================

class ReshapeAliasExpandModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.reshape_alias = torch.ops.aten._reshape_alias

    @export
    @annotate_args([
        None,
        ([-1], torch.float32, True),
    ])

    def forward(self, a):
        return torch.ops.aten._reshape_alias(a, size=(12, 32), stride=(32, 1))

@register_test_case(module_factory=lambda: ReshapeAliasExpandModule())
def ReshapeAliasExpandModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(384))

# ==============================================================================

class ReshapeAliasCollapseModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])

    def forward(self, a):
        return torch.ops.aten._reshape_alias(a, (8,), (1,))

@register_test_case(module_factory=lambda: ReshapeAliasCollapseModule())
def ReshapeAliasCollapseModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4))