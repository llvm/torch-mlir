# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================


class IndexSelectSingleIdxModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([4, 5, 6], torch.float32, True),
        ([1], torch.int64, True),
    ])

    def forward(self, input, indices):
        return torch.index_select(input, 1, indices)

@register_test_case(module_factory=lambda: IndexSelectSingleIdxModule())
def IndexSelectSingleIdxModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(4, 5, 6), torch.tensor([2]))


class IndexSelectTwoIdxModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([4, 5, 6], torch.float32, True),
        ([2], torch.int64, True),
    ])

    def forward(self, input, indices):
        return torch.index_select(input, 2, indices)

@register_test_case(module_factory=lambda: IndexSelectTwoIdxModule())
def IndexSelectTwoIdxModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(4, 5, 6), torch.tensor([2, 4]))


class IndexSelectWholeDimensionModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([4, 5, 6], torch.float32, True),
        ([4], torch.int64, True),
    ])

    def forward(self, input, indices):
        return torch.index_select(input, 0, indices)

@register_test_case(module_factory=lambda: IndexSelectWholeDimensionModule())
def IndexSelectWholeDimensionModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(4, 5, 6), torch.tensor([0, 1, 2, 3]))


class IndexSelectWholeTensorModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([3], torch.float32, True),
        ([3], torch.int64, True),
    ])

    def forward(self, input, indices):
        return torch.index_select(input, 0, indices)

@register_test_case(module_factory=lambda: IndexSelectWholeTensorModule())
def IndexSelectWholeTensorModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(3), torch.tensor([0, 1, 2]))


class IndexSelectDynamicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1], torch.int64, True),
    ])

    def forward(self, input, indices):
        return torch.index_select(input, 2, indices)

@register_test_case(module_factory=lambda: IndexSelectDynamicModule())
def IndexSelectDynamicModulebasic(module, tu: TestUtils):
    module.forward(torch.randn(4, 5, 6), torch.tensor([0, 4]))


class IndexSelectDynamicInputSizeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([2], torch.int64, True),
    ])

    def forward(self, input, indices):
        return torch.index_select(input, 2, indices)

@register_test_case(module_factory=lambda: IndexSelectDynamicInputSizeModule())
def IndexSelectDynamicInputSizeModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(4, 5, 6), torch.tensor([0, 2]))


class IndexSelectDynamicIndexSizeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([4, 5, 6], torch.float32, True),
        ([-1], torch.int64, True),
    ])

    def forward(self, input, indices):
        return torch.index_select(input, 1, indices)

@register_test_case(module_factory=lambda: IndexSelectDynamicIndexSizeModule())
def IndexSelectDynamicIndexSizeModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(4, 5, 6), torch.tensor([1, 2]))
