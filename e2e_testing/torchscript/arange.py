# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.torchscript.framework import TestUtils
from torch_mlir_e2e_test.torchscript.registry import register_test_case
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export

# ==============================================================================


class ArangeIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])

    def forward(self):
        return torch.arange(5)

@register_test_case(module_factory=lambda: ArangeIntModule())
def ArangeIntModule_basic(module, tu: TestUtils):
    module.forward()


class ArangeFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])

    def forward(self):
        return torch.arange(5.0)

@register_test_case(module_factory=lambda: ArangeFloatModule())
def ArangeFloatModule_basic(module, tu: TestUtils):
    module.forward()


class ArangeZeroElementOutputModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])

    def forward(self):
        return torch.arange(0)

@register_test_case(module_factory=lambda: ArangeZeroElementOutputModule())
def ArangeZeroElementOutputModule_basic(module, tu: TestUtils):
    module.forward()


class ArangeStartIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])

    def forward(self):
        return torch.arange(0, 5)

@register_test_case(module_factory=lambda: ArangeStartIntModule())
def ArangeStartIntModule_basic(module, tu: TestUtils):
    module.forward()


class ArangeStartFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])

    def forward(self):
        return torch.arange(0.0, 5.0)

@register_test_case(module_factory=lambda: ArangeStartFloatModule())
def ArangeStartFloatModule_basic(module, tu: TestUtils):
    module.forward()


class ArangeNegativeStartIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])

    def forward(self):
        return torch.arange(-10, 5)

@register_test_case(module_factory=lambda: ArangeNegativeStartIntModule())
def ArangeNegativeStartIntModule_basic(module, tu: TestUtils):
    module.forward()


class ArangeNegativeStartFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])

    def forward(self):
        return torch.arange(-1.0, 5.0)

@register_test_case(module_factory=lambda: ArangeNegativeStartFloatModule())
def ArangeNegativeStartFloatModule_basic(module, tu: TestUtils):
    module.forward()


class ArangeStartStepIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])

    def forward(self):
        return torch.arange(0, 5, 1)

@register_test_case(module_factory=lambda: ArangeStartStepIntModule())
def ArangeStartStepIntModule_basic(module, tu: TestUtils):
    module.forward()


class ArangeStartStepFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])

    def forward(self):
        return torch.arange(-1, 5, 1.0)

@register_test_case(module_factory=lambda: ArangeStartStepFloatModule())
def ArangeStartStepFloatModule_basic(module, tu: TestUtils):
    module.forward()


class ArangeStartNegativeStepIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])

    def forward(self):
        return torch.arange(10, 1, -2)

@register_test_case(module_factory=lambda: ArangeStartNegativeStepIntModule())
def ArangeStartNegativeStepIntModule_basic(module, tu: TestUtils):
    module.forward()


class ArangeStartNegativeStepFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])

    def forward(self):
        return torch.arange(-1, -15, -3.0)

@register_test_case(module_factory=lambda: ArangeStartNegativeStepFloatModule())
def ArangeStartNegativeStepFloatModule_basic(module, tu: TestUtils):
    module.forward()
