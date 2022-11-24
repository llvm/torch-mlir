# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================


class SqueezeStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([1, 7, 1, 3, 1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.squeeze(a)


@register_test_case(
    module_factory=lambda: SqueezeStaticModule())
def SqueezeModule_static(module, tu: TestUtils):
    module.forward(tu.rand(1, 7, 1, 3, 1))


# ==============================================================================


class SqueezeAllUnitDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([1, 1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.squeeze(a)


@register_test_case(
    module_factory=lambda: SqueezeAllUnitDimModule())
def SqueezeModule_allUnitDim(module, tu: TestUtils):
    module.forward(tu.rand(1, 1))


# ==============================================================================


class SqueezeBroadcastModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([], torch.float32, True),
    ])
    def forward(self, a, b):
        return a * b.squeeze()


@register_test_case(
    module_factory=lambda: SqueezeBroadcastModule())
def SqueezeModule_broadcast(module, tu: TestUtils):
    module.forward(tu.rand(4, 3), tu.rand())


# ==============================================================================


class SqueezeDimStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([1, 7], torch.float32, True),
    ])
    def forward(self, a):
        return torch.squeeze(a, 0)


@register_test_case(
    module_factory=lambda: SqueezeDimStaticModule())
def SqueezeDimModule_static(module, tu: TestUtils):
    module.forward(tu.rand(1, 7))
    

# ==============================================================================


class SqueezeDimDynamicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, 1, 384, -1, 1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.squeeze(a, 4)


@register_test_case(
    module_factory=lambda: SqueezeDimDynamicModule())
def SqueezeDimModule_dynamic(module, tu: TestUtils):
    module.forward(tu.rand(8, 1, 384, 12, 1))


# ==============================================================================


class SqueezeDimNegDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([1, -1, 1, 384, -1, 1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.squeeze(a, -6)


@register_test_case(
    module_factory=lambda: SqueezeDimNegDimModule())
def SqueezeDimModule_negDim(module, tu: TestUtils):
    module.forward(tu.rand(1, 8, 1, 384, 12, 1))


# ==============================================================================


class SqueezeDimIdentityModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([4, 1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.squeeze(a, 0)


@register_test_case(
    module_factory=lambda: SqueezeDimIdentityModule())
def SqueezeDimModule_identity(module, tu: TestUtils):
    module.forward(tu.rand(4, 1, 3))


# ==============================================================================


class SqueezeDimUnitDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.squeeze(a, 0)


@register_test_case(
    module_factory=lambda: SqueezeDimUnitDimModule())
def SqueezeDimModule_unitDim(module, tu: TestUtils):
    module.forward(tu.rand(1))
