#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================


class DiagModule1D(torch.nn.Module):
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
        return torch.ops.aten.diag(a)


@register_test_case(module_factory=lambda: DiagModule1D())
def DiagModule1D_basic(module, tu: TestUtils):
    module.forward(tu.rand(3))


# ==============================================================================


class DiagModule1DPositiveOffset(torch.nn.Module):
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
        return torch.ops.aten.diag(a, 1)


@register_test_case(module_factory=lambda: DiagModule1DPositiveOffset())
def DiagModule1D_positive_offset(module, tu: TestUtils):
    module.forward(tu.rand(3))


# ==============================================================================


class DiagModule1DNegativeOffset(torch.nn.Module):
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
        return torch.ops.aten.diag(a, -1)


@register_test_case(module_factory=lambda: DiagModule1DNegativeOffset())
def DiagModule1D_negative_offset(module, tu: TestUtils):
    module.forward(tu.rand(3))


# ==============================================================================


class DiagModule2D(torch.nn.Module):
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
        return torch.ops.aten.diag(a)


@register_test_case(module_factory=lambda: DiagModule2D())
def DiagModule2D_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 3))


@register_test_case(module_factory=lambda: DiagModule2D())
def DiagModule2D_nonsquare(module, tu: TestUtils):
    module.forward(tu.rand(3, 5))


# ==============================================================================


class DiagModule2DPositiveOffset(torch.nn.Module):
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
        return torch.ops.aten.diag(a, 1)


@register_test_case(module_factory=lambda: DiagModule2DPositiveOffset())
def DiagModule2D_positive_offset(module, tu: TestUtils):
    module.forward(tu.rand(4, 4))


# ==============================================================================


class DiagModule2DNegativeOffset(torch.nn.Module):
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
        return torch.ops.aten.diag(a, -1)


@register_test_case(module_factory=lambda: DiagModule2DNegativeOffset())
def DiagModule2D_negative_offset(module, tu: TestUtils):
    module.forward(tu.rand(4, 4))
