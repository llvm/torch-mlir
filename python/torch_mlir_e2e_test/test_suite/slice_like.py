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
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return x[0:5:1, 1:3:1, 2:4:1]


@register_test_case(module_factory=lambda: SliceModule())
def SliceModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6,4,7))


# ==============================================================================

class SliceStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([6, 4, 7], torch.float32, True),
    ])
    def forward(self, x):
        return x[0:5:1, 1:3:1, 2:4:1]


@register_test_case(module_factory=lambda: SliceStaticModule())
def SliceStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6,4,7))


# ==============================================================================

class SliceOutOfUpperBoundIndexModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        # TODO: remove hacky cat tensor once refbackend supports 0 size dim
        result =  x[:8, :5, 8:]
        cat_tensor = torch.ones((6,4,1), dtype=torch.float32)
        return torch.cat((result,cat_tensor), dim=2)


@register_test_case(module_factory=lambda: SliceOutOfUpperBoundIndexModule())
def SliceOutOfUpperBoundIndexModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6,4,7))

# ==============================================================================

class SliceOutOfLowerBoundEndIndexModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return x[:-8,-7:,:]


@register_test_case(module_factory=lambda: SliceOutOfLowerBoundEndIndexModule())
def SliceOutOfLowerBoundEndIndexModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6,4,7))

# ==============================================================================

class SliceOutOfLowerBoundStartIndexModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return x[-8:3:1, 1:3:1, 2:4:1]


@register_test_case(module_factory=lambda: SliceOutOfLowerBoundStartIndexModule())
def SliceOutOfLowerBoundStartIndexModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6,4,7))

# ==============================================================================


class SliceEndSleStartModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        # TODO: remove hacky cat tensor once refbackend supports 0 size dim
        result = x[:, 4:3, :]
        cat_tensor = torch.ones((6,1,7), dtype=torch.float32)
        return torch.cat((result, cat_tensor), dim=1)


@register_test_case(module_factory=lambda: SliceEndSleStartModule())
def SliceEndSleStartModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6,4,7))

# ==============================================================================


class SliceStartEqEndModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        # TODO: remove hacky cat tensor once refbackend supports 0 size dim
        result = x[5:5, :, :]
        cat_tensor = torch.ones((1,4,7), dtype=torch.float32)
        return torch.cat((result, cat_tensor), dim=0)


@register_test_case(module_factory=lambda: SliceStartEqEndModule())
def SliceStartEqEndModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6,4,7))

# ==============================================================================

class SliceSizeTwoStepModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return x[0:5:2, 0:3:2, 0:4:2]


@register_test_case(module_factory=lambda: SliceSizeTwoStepModule())
def SliceSizeTwoStepModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(10,5,17))

# ==============================================================================

class SliceNegIdxModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return x[:-1, -2:-1]


@register_test_case(module_factory=lambda: SliceNegIdxModule())
def SliceNegIdxModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3,9))

# ==============================================================================

class SliceSingleIdxModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return x[0]


@register_test_case(module_factory=lambda: SliceSingleIdxModule())
def SliceSingleIdxModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6,8))

# ==============================================================================

class SliceWholeTensorModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return x[:, :]


@register_test_case(module_factory=lambda: SliceWholeTensorModule())
def SliceWholeTensorModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6,8))

# ==============================================================================

class SelectIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, x):
        return x.select(0,0)


@register_test_case(module_factory=lambda: SelectIntModule())
def SelectIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(5,5, high=10))

# ==============================================================================

# For aten.slice_scatter op, The arguments are: SliceScatter(input, src, dim=0, start=None, end=None, step=1).
# For aten.select_scatter op, The arguments are: SelectScatter(input, src, dim=0, index).
class SliceScatterModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x, src):
        return torch.ops.aten.slice_scatter(x, src, dim = 1, start = 0, end = 1, step = 1)

@register_test_case(module_factory=lambda: SliceScatterModule())
def SliceScatterModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6, 8), tu.rand(6, 1))

class SliceScatterZeroDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x, src):
        return torch.ops.aten.slice_scatter(x, src, dim = 0, start = 0, end = 1, step = 1)


@register_test_case(module_factory=lambda: SliceScatterZeroDimModule())
def SliceScatterZeroDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6, 8), tu.rand(1, 8))


class SliceScatterNegativeDimModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x, src):
        return torch.ops.aten.slice_scatter(x,
                                            src,
                                            dim=-2,
                                            start=0,
                                            end=1,
                                            step=1)


@register_test_case(module_factory=lambda: SliceScatterNegativeDimModule())
def SliceScatterNegativeDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6, 8), tu.rand(1, 8))

class SliceScatterStepVariationModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x, src):
        return torch.ops.aten.slice_scatter(x, src, dim = 1, start = 0, end = 1, step = 2)


@register_test_case(module_factory=lambda: SliceScatterStepVariationModule())
def SliceScatterStepVariationModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6, 8), tu.rand(6, 1))

class SliceScatterStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([6, 8], torch.float32, True),
        ([6, 1], torch.float32, True),
    ])
    def forward(self, x, src):
        return torch.ops.aten.slice_scatter(x, src, dim = 1, start = 0, end = 1, step = 1)


@register_test_case(module_factory=lambda: SliceScatterStaticModule())
def SliceScatterStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6, 8), tu.rand(6, 1))

class SelectScatterModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x, src):
        return torch.ops.aten.select_scatter(x, src, dim = 0, index = 0)


@register_test_case(module_factory=lambda: SelectScatterModule())
def SelectScattertModule_basic(module, tu: TestUtils):
    module.forward(torch.rand(6, 8, 5), torch.rand(8, 5))

class SelectScatterStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([6, 8, 5], torch.float32, True),
        ([6, 5], torch.float32, True),
    ])
    def forward(self, x, src):
        return torch.ops.aten.select_scatter(x, src, dim = 1, index = 0)


@register_test_case(module_factory=lambda: SelectScatterStaticModule())
def SelectScattertStaticModule_basic(module, tu: TestUtils):
    module.forward(torch.rand(6, 8, 5), torch.rand(6, 5))

# ==============================================================================

class NarrowHorizontalTest(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.narrow(x, dim=0, start=0, length=2)
        

@register_test_case(module_factory=lambda: NarrowHorizontalTest())
def NarrowHorizontalTest_basic(module, tu: TestUtils):
    module.forward(tu.rand(6,4,3))

# ==============================================================================


class NarrowVerticalTest(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.narrow(x, dim=1, start=0, length=2)


@register_test_case(module_factory=lambda: NarrowVerticalTest())
def NarrowVerticalTest_basic(module, tu: TestUtils):
    module.forward(tu.rand(6,4,3))

# ==============================================================================

class NarrowHorizontalTest2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.narrow(x, dim=0, start=0, length=2)
        

@register_test_case(module_factory=lambda: NarrowHorizontalTest2())
def NarrowHorizontalTest2_basic(module, tu: TestUtils):
    module.forward(tu.rand(6,4))

# ==============================================================================


class NarrowVerticalTest2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.narrow(x, dim=1, start=0, length=2)


@register_test_case(module_factory=lambda: NarrowVerticalTest2())
def NarrowVerticalTest2_basic(module, tu: TestUtils):
    module.forward(tu.rand(6,4))
