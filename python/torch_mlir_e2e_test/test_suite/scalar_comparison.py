# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================


class NeIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([], torch.int64, True),
        ([], torch.int64, True),
    ])
    def forward(self, lhs, rhs):
        return int(lhs) != int(rhs)


@register_test_case(module_factory=lambda: NeIntModule())
def NeIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(low=-100, high=100), tu.randint(low=-100, high=100))


# ==============================================================================


class EqIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([], torch.int64, True),
        ([], torch.int64, True),
    ])
    def forward(self, lhs, rhs):
        return int(lhs) == int(rhs)


@register_test_case(module_factory=lambda: EqIntModule())
def EqIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(low=-100, high=100), tu.randint(low=-100, high=100))


# ==============================================================================


class GtIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([], torch.int64, True),
        ([], torch.int64, True),
    ])
    def forward(self, lhs, rhs):
        return int(lhs) > int(rhs)


@register_test_case(module_factory=lambda: GtIntModule())
def GtIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(low=-100, high=100), tu.randint(low=-100, high=100))


# ==============================================================================


class GeIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([], torch.int64, True),
        ([], torch.int64, True),
    ])
    def forward(self, lhs, rhs):
        return torch.ops.aten.ge(int(lhs), int(rhs))


@register_test_case(module_factory=lambda: GeIntModule())
def GeIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(low=-100, high=100), tu.randint(low=-100, high=100))


# ==============================================================================


class GeFloatModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([], torch.float64, True),
        ([], torch.float64, True),
    ])
    def forward(self, lhs, rhs):
        return float(lhs) >= float(rhs)


@register_test_case(module_factory=lambda: GeFloatModule())
def GeFloatModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(()).double(), torch.randn(()).double())


# ==============================================================================


class GeFloatIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([], torch.float64, True),
        ([], torch.int64, True),
    ])
    def forward(self, lhs, rhs):
        return float(lhs) >= int(rhs)


@register_test_case(module_factory=lambda: GeFloatIntModule())
def GeFloatIntModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(()).double(), tu.randint(low=-100, high=100))


# ==============================================================================


class NeFloatIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([], torch.float64, True),
        ([], torch.int64, True),
    ])
    def forward(self, lhs, rhs):
        return float(lhs) != int(rhs)


@register_test_case(module_factory=lambda: NeFloatIntModule())
def NeFloatIntModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(()).double(), tu.randint(low=-100, high=100))


# ==============================================================================


class GtFloatIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([], torch.float64, True),
        ([], torch.int64, True),
    ])
    def forward(self, lhs, rhs):
        return float(lhs) > int(rhs)


@register_test_case(module_factory=lambda: GtFloatIntModule())
def GtFloatIntModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(()).double(), tu.randint(low=-100, high=100))
