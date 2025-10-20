# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================


class KlDivLossModule_default(torch.nn.Module):
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
        return torch.ops.aten.kl_div(x, y)


@register_test_case(module_factory=lambda: KlDivLossModule_default())
def KlDivLossModule_default_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5, 2), tu.rand(3, 5, 2))


# ==============================================================================


class KlDivLossModule_reduction_is_none(torch.nn.Module):
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
        return torch.ops.aten.kl_div(x, y, reduction=0)


@register_test_case(module_factory=lambda: KlDivLossModule_reduction_is_none())
def KlDivLossModule_reduction_is_none_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5, 2), tu.rand(3, 5, 2))


# ==============================================================================


class KlDivLossModule_reduction_is_none_log_target_is_true(torch.nn.Module):
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
        return torch.ops.aten.kl_div(x, y, reduction=0, log_target=True)


@register_test_case(
    module_factory=lambda: KlDivLossModule_reduction_is_none_log_target_is_true()
)
def KlDivLossModule_reduction_is_none_log_target_is_true_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5, 2), tu.rand(3, 5, 2))


# ==============================================================================


class KlDivLossModule_mean_reduction(torch.nn.Module):
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
        return torch.ops.aten.kl_div(x, y, reduction=1)


@register_test_case(module_factory=lambda: KlDivLossModule_mean_reduction())
def KlDivLossModule_mean_reduction_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5, 2), tu.rand(3, 5, 2))


# ==============================================================================


class KlDivLossModule_sum_reduction(torch.nn.Module):
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
        return torch.ops.aten.kl_div(x, y, reduction=2)


@register_test_case(module_factory=lambda: KlDivLossModule_sum_reduction())
def KlDivLossModule_sum_reduction_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5, 2), tu.rand(3, 5, 2))


# ==============================================================================


class KlDivLossModule_batchmean_reduction(torch.nn.Module):
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
    def forward(self, input, target):
        # torch.ops.aten.kl_div has no direct way to pass batchmean as reduction mode.
        # https://github.com/pytorch/pytorch/blob/53ecb8159aa28b3c015917acaa89604cfae0d2c6/torch/nn/_reduction.py#L8-L24
        # F.kl_div(input, target, reduction="batchmean"):
        # out = torch.kl_div(input, target, reduction="sum")
        # batch_size = input.shape[0]
        # out = out / batch_size
        # https://github.com/pytorch/pytorch/blob/53ecb8159aa28b3c015917acaa89604cfae0d2c6/torch/nn/functional.py#L3379-L3381
        loss = torch.ops.aten.kl_div(input, target, reduction=2)
        batch_size = input.shape[0]
        return torch.ops.aten.div.Scalar(loss, batch_size)


@register_test_case(module_factory=lambda: KlDivLossModule_batchmean_reduction())
def KlDivLossModule_batchmean_reduction_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5, 2), tu.rand(3, 5, 2))


# ==============================================================================
