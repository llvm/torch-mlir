# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export


# ==============================================================================


class PdistForwardModuleP2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, input):
        return torch.ops.aten._pdist_forward(input, 2)


@register_test_case(module_factory=lambda: PdistForwardModuleP2())
def PdistForwardModuleP2_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 8))


# ==============================================================================


class PdistForwardModuleP1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, input):
        return torch.ops.aten._pdist_forward(input, 1)


@register_test_case(module_factory=lambda: PdistForwardModuleP1())
def PdistForwardModuleP1_basic(module, tu: TestUtils):
    module.forward(tu.rand(6, 4))


# ==============================================================================


class CdistForwardModuleP2CNone(torch.nn.Module):
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
    def forward(self, x1, x2):
        return torch.ops.aten._cdist_forward(x1, x2, 2, None)


@register_test_case(module_factory=lambda: CdistForwardModuleP2CNone())
def CdistForwardModuleP2CNone_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5, 8), tu.rand(3, 7, 8))


# ==============================================================================


class CdistForwardModuleP1C2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x1, x2):
        return torch.ops.aten._cdist_forward(x1, x2, 1, 2)


@register_test_case(module_factory=lambda: CdistForwardModuleP1C2())
def CdistForwardModuleP1C2_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4, 6), tu.rand(2, 3, 6, 6))


# ==============================================================================


class CdistForwardModuleP2C1(torch.nn.Module):
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
    def forward(self, x1, x2):
        return torch.ops.aten._cdist_forward(x1, x2, 2, 1)


@register_test_case(module_factory=lambda: CdistForwardModuleP2C1())
def CdistForwardModuleP2C1_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4, 6), tu.rand(2, 6, 6))
