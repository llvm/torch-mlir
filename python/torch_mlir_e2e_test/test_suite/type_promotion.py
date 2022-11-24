# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================


class TypePromotionSameCategoryDifferentWidthModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.int32, True),
        ([-1], torch.int64, True),
    ])
    def forward(self, a, b):
        return torch.add(a, b, alpha=3)


@register_test_case(
    module_factory=lambda: TypePromotionSameCategoryDifferentWidthModule())
def TypePromotionSameCategoryDifferentWidthModule_basic(module, tu: TestUtils):
    module.forward(
        tu.randint(4, high=10).type(torch.int32),
        tu.randint(4, high=10))


class TypePromotionDifferentCategoryModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.int64, True),
        ([-1], torch.float32, True),
    ])
    def forward(self, a, b):
        return torch.add(a, b, alpha=3)


@register_test_case(
    module_factory=lambda: TypePromotionDifferentCategoryModule())
def TypePromotionDifferentCategoryModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(4, high=10), torch.randn(4))


class TypePromotionSameCategoryZeroRankWiderModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.float32, True),
        ([], torch.float64, True),
    ])
    def forward(self, a, b):
        return torch.add(a, b, alpha=2.3)


@register_test_case(
    module_factory=lambda: TypePromotionSameCategoryZeroRankWiderModule())
def TypePromotionSameCategoryZeroRankWider_basic(module, tu: TestUtils):
    module.forward(tu.rand(4), tu.rand().type(torch.float64))


class TypePromotionZeroRankHigherCategoryModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.int64, True),
        ([], torch.float32, True),
    ])
    def forward(self, a, b):
        return torch.add(a, b, alpha=2)


@register_test_case(
    module_factory=lambda: TypePromotionZeroRankHigherCategoryModule())
def TypePromotionZeroRankHigherCategoryModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(4, high=10), tu.rand())


class TypePromotionAlphaWiderModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.float32, True),
        ([], torch.float32, True),
    ])
    def forward(self, a, b):
        return torch.add(a, b, alpha=2.3)


@register_test_case(module_factory=lambda: TypePromotionAlphaWiderModule())
def TypePromotionAlphaWiderModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4), tu.rand())
