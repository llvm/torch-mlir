# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================


class HuberLossModule_default(torch.nn.Module):
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
        return torch.ops.aten.huber_loss(x, y)


@register_test_case(module_factory=lambda: HuberLossModule_default())
def HuberLossModule_default_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5, 2), tu.rand(3, 5, 2))


# ==============================================================================


class HuberLossModule_reduction_is_none(torch.nn.Module):
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
        return torch.ops.aten.huber_loss(x, y, delta=2.3, reduction=0)


@register_test_case(module_factory=lambda: HuberLossModule_reduction_is_none())
def HuberLossModule_reduction_is_none_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5, 2), tu.rand(3, 5, 2))


# ==============================================================================


class HuberLossModule_mean_reduction(torch.nn.Module):
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
        return torch.ops.aten.huber_loss(x, y, reduction=1)


@register_test_case(module_factory=lambda: HuberLossModule_mean_reduction())
def HuberLossModule_mean_reduction_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5, 2), tu.rand(3, 5, 2))


# ==============================================================================


class HuberLossModule_sum_reduction(torch.nn.Module):
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
        return torch.ops.aten.huber_loss(x, y, reduction=2)


@register_test_case(module_factory=lambda: HuberLossModule_sum_reduction())
def HuberLossModule_sum_reduction_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5, 2), tu.rand(3, 5, 2))


# ==============================================================================
