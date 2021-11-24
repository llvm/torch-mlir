# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.torchscript.framework import TestUtils
from torch_mlir_e2e_test.torchscript.registry import register_test_case
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export

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


class SqueezeDynamicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([1, -1, 1, 384, -1, 1, 1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.squeeze(a)


@register_test_case(
    module_factory=lambda: SqueezeDynamicModule())
def SqueezeModule_dynamic(module, tu: TestUtils):
    module.forward(tu.rand(1, 8, 1, 384, 12, 1, 1))


# ==============================================================================


class SqueezeNoUnitDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([4, -1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.squeeze(a)


@register_test_case(
    module_factory=lambda: SqueezeNoUnitDimModule())
def SqueezeModule_noUnitDim(module, tu: TestUtils):
    module.forward(tu.rand(4, 2, 3))


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

