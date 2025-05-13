# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import functorch
import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================


class ReflectionPad2dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1, 20, 20], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.reflection_pad2d(x, (10, 10, 10, 10))


@register_test_case(module_factory=lambda: ReflectionPad2dModule())
def ReflectionPad2dModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 20, 20, low=-1))


# ==============================================================================


class ReflectionPad2dModuleTop(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1, 3, 4], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.reflection_pad2d(x, (0, 0, 2, 0))


@register_test_case(module_factory=lambda: ReflectionPad2dModuleTop())
def ReflectionPad2dModule_Top(module, tu: TestUtils):
    module.forward(tu.rand(1, 3, 4))


# ==============================================================================


class ReflectionPad2dModuleBottom(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 3, 10, 10], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.reflection_pad2d(x, (0, 0, 0, 5))


@register_test_case(module_factory=lambda: ReflectionPad2dModuleBottom())
def ReflectionPad2dModule_Bottom(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 10, 10))


# ==============================================================================


class ReflectionPad2dModuleLeft(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 3, 20, 20], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.reflection_pad2d(x, (15, 0, 0, 0))


@register_test_case(module_factory=lambda: ReflectionPad2dModuleLeft())
def ReflectionPad2dModule_Left(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 20, 20))


# ==============================================================================


class ReflectionPad2dModuleRight(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 3, 20, 20], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.reflection_pad2d(x, (0, 11, 0, 0))


@register_test_case(module_factory=lambda: ReflectionPad2dModuleRight())
def ReflectionPad2dModule_Right(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 20, 20))


# ==============================================================================


class ReflectionPad3dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1, 20, 20, 20, 20], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.reflection_pad3d(x, (10, 10, 10, 10, 10, 10))


@register_test_case(module_factory=lambda: ReflectionPad3dModule())
def ReflectionPad3dModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 20, 20, 20, 20, low=-1))


# ==============================================================================


class ReflectionPad3dModuleTop(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1, 3, 4, 5, 6], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.reflection_pad3d(x, (0, 0, 2, 0, 0, 0))


@register_test_case(module_factory=lambda: ReflectionPad3dModuleTop())
def ReflectionPad3dModuleTop_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 3, 4, 5, 6))


# ==============================================================================


class ReflectionPad3dModuleBottom(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 3, 10, 10, 6], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.reflection_pad3d(x, (0, 0, 0, 5, 0, 0))


@register_test_case(module_factory=lambda: ReflectionPad3dModuleBottom())
def ReflectionPad3dModuleBottom_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 10, 10, 6))


# ==============================================================================


class ReflectionPad3dModuleLeft(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 3, 20, 20, 10], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.reflection_pad3d(x, (9, 0, 0, 0, 0, 0))


@register_test_case(module_factory=lambda: ReflectionPad3dModuleLeft())
def ReflectionPad3dModuleLeft_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 20, 20, 10))


# ==============================================================================


class ReflectionPad3dModuleRight(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 3, 20, 20, 12], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.reflection_pad3d(x, (0, 11, 0, 0, 0, 0))


@register_test_case(module_factory=lambda: ReflectionPad3dModuleRight())
def ReflectionPad3dModuleRight_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 20, 20, 12))


# ==============================================================================


class ReflectionPad3dModuleFront(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 3, 20, 20, 12], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.reflection_pad3d(x, (0, 0, 0, 0, 5, 0))


@register_test_case(module_factory=lambda: ReflectionPad3dModuleFront())
def ReflectionPad3dModuleFront_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 20, 20, 12))


# ==============================================================================


class ReflectionPad3dModuleBack(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 3, 20, 20, 12], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.reflection_pad3d(x, (0, 0, 0, 0, 0, 7))


@register_test_case(module_factory=lambda: ReflectionPad3dModuleBack())
def ReflectionPad3dModuleBack_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 20, 20, 12))
