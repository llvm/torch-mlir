# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.torchscript.framework import TestUtils
from torch_mlir_e2e_test.torchscript.registry import register_test_case
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export

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
    module.forward(torch.randint(10, (5,5)))

# ==============================================================================


class SplitModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        split_sizes = [1, 3, 5]
        dim = 0
        splits = torch.ops.aten.split(x, split_sizes, dim)
        return splits[0], splits[1], splits[2]

@register_test_case(module_factory=lambda: SplitModule())
def SplitModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(9, 6, 8))

# ==============================================================================

class SplitStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([9, 6, 8], torch.float32, True),
    ])
    def forward(self, x):
        split_sizes = [1, 3, 2]
        dim = 1
        splits = torch.ops.aten.split(x, split_sizes, dim)
        return splits[0], splits[1], splits[2]

@register_test_case(module_factory=lambda: SplitStaticModule())
def SplitStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(9, 6, 8))

# ==============================================================================

class SplitWithSizesModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        split_sizes = [1, 3, 4]
        dim = 2
        splits = torch.ops.aten.split_with_sizes(x, split_sizes, dim)
        return splits[0], splits[1], splits[2]

@register_test_case(module_factory=lambda: SplitWithSizesModule())
def SplitWithSizestModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(9, 6, 8))

# ==============================================================================

class SplitTensorModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        split_size = 2
        dim = 2
        splits = torch.ops.aten.split(x.view(9, 6, 8), split_size, dim)
        return splits[0], splits[1], splits[2], splits[3]

@register_test_case(module_factory=lambda: SplitTensorModule())
def SplitTensorModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(9, 6, 8))

class ChunkModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        chunks = 4
        dim = 2
        splits = torch.ops.aten.chunk(x.view(9, 6, 8), chunks, dim)
        return splits[0], splits[1], splits[2], splits[3]

@register_test_case(module_factory=lambda: ChunkModule())
def ChunkModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(9, 6, 8))