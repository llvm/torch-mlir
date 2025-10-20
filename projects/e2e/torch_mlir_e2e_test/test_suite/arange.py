# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================


class ArangeIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        return torch.arange(5)


@register_test_case(module_factory=lambda: ArangeIntModule())
def ArangeIntModule_basic(module, tu: TestUtils):
    module.forward()


class ArangeFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        return torch.arange(5.0)


@register_test_case(module_factory=lambda: ArangeFloatModule())
def ArangeFloatModule_basic(module, tu: TestUtils):
    module.forward()


class ArangeZeroElementOutputModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        return torch.arange(0)


@register_test_case(module_factory=lambda: ArangeZeroElementOutputModule())
def ArangeZeroElementOutputModule_basic(module, tu: TestUtils):
    module.forward()


# ==============================================================================


class ArangeStartIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        return torch.arange(0, 5)


@register_test_case(module_factory=lambda: ArangeStartIntModule())
def ArangeStartIntModule_basic(module, tu: TestUtils):
    module.forward()


class ArangeStartFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        return torch.arange(0.0, 5.0)


@register_test_case(module_factory=lambda: ArangeStartFloatModule())
def ArangeStartFloatModule_basic(module, tu: TestUtils):
    module.forward()


class ArangeNegativeStartIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        return torch.arange(-10, 5)


@register_test_case(module_factory=lambda: ArangeNegativeStartIntModule())
def ArangeNegativeStartIntModule_basic(module, tu: TestUtils):
    module.forward()


class ArangeNegativeStartFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        return torch.arange(-1.4, 5.7)


@register_test_case(module_factory=lambda: ArangeNegativeStartFloatModule())
def ArangeNegativeStartFloatModule_basic(module, tu: TestUtils):
    module.forward()


# ==============================================================================


class ArangeStartStepIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        return torch.arange(0, 5, 1)


@register_test_case(module_factory=lambda: ArangeStartStepIntModule())
def ArangeStartStepIntModule_basic(module, tu: TestUtils):
    module.forward()


class ArangeStartStepFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        return torch.arange(-1, 5, 1.3)


@register_test_case(module_factory=lambda: ArangeStartStepFloatModule())
def ArangeStartStepFloatModule_basic(module, tu: TestUtils):
    module.forward()


class ArangeStartNegativeStepIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        return torch.arange(10, 1, -2)


@register_test_case(module_factory=lambda: ArangeStartNegativeStepIntModule())
def ArangeStartNegativeStepIntModule_basic(module, tu: TestUtils):
    module.forward()


class ArangeStartNegativeStepFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        return torch.arange(-1, -15, -3.4)


@register_test_case(module_factory=lambda: ArangeStartNegativeStepFloatModule())
def ArangeStartNegativeStepFloatModule_basic(module, tu: TestUtils):
    module.forward()


# ==============================================================================


class ArangeDtypeFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        return torch.arange(-1, 15, dtype=torch.float32)


@register_test_case(module_factory=lambda: ArangeDtypeFloatModule())
def ArangeDtypeFloatModule_basic(module, tu: TestUtils):
    module.forward()


class ArangeDtypeIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        return torch.arange(0.2, 5.0, dtype=torch.int64)


@register_test_case(module_factory=lambda: ArangeDtypeIntModule())
def ArangeDtypeIntModule_basic(module, tu: TestUtils):
    module.forward()


# ==============================================================================


class ArangeFalsePinMemoryModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        return torch.arange(5.0, dtype=torch.int64, pin_memory=False)


@register_test_case(module_factory=lambda: ArangeFalsePinMemoryModule())
def ArangeFalsePinMemoryModule_basic(module, tu: TestUtils):
    module.forward()


# ==============================================================================


class ArangeStartOutModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([12], torch.int64, True),
        ]
    )
    def forward(self, x):
        return torch.arange(start=0, end=12, out=x)


@register_test_case(module_factory=lambda: ArangeStartOutModule())
def ArangeStartOutModule_basic(module, tu: TestUtils):
    module.forward(torch.zeros(12).to(torch.int64))


class ArangeStartOutViewModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([3, 4], torch.int64, True),
        ]
    )
    def forward(self, x):
        return torch.arange(start=1, end=13, out=x)


@register_test_case(module_factory=lambda: ArangeStartOutViewModule())
def ArangeStartOutViewModule_basic(module, tu: TestUtils):
    module.forward(torch.zeros(3, 4).to(torch.int64))


class ArangeStartOutDtypeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([12], torch.int64, True),
        ]
    )
    def forward(self, x):
        return torch.arange(start=1.1, end=13.1, out=x)


@register_test_case(module_factory=lambda: ArangeStartOutDtypeModule())
def ArangeStartOutDtypeModule_basic(module, tu: TestUtils):
    module.forward(torch.zeros(12).to(torch.int64))


# ==============================================================================


class LinspaceModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        return torch.linspace(-10.1, 10.1, 10)


@register_test_case(module_factory=lambda: LinspaceModule())
def LinspaceModule_basic(module, tu: TestUtils):
    module.forward()


class LinspaceDtypeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        return torch.linspace(-10.1, 10.1, 10, dtype=torch.int64)


@register_test_case(module_factory=lambda: LinspaceDtypeModule())
def LinspaceDtypeModule_basic(module, tu: TestUtils):
    module.forward()


class LinspaceEmptyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        return torch.linspace(-10.1, 10.1, 0)


@register_test_case(module_factory=lambda: LinspaceEmptyModule())
def LinspaceEmptyModule_basic(module, tu: TestUtils):
    module.forward()


class LinspaceOneSizeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        return torch.linspace(-10.1, 10.1, 1)


@register_test_case(module_factory=lambda: LinspaceOneSizeModule())
def LinspaceOneSizeModule_basic(module, tu: TestUtils):
    module.forward()


class LinspaceTwoSizeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        return torch.linspace(-10.1, 10.1, 2)


@register_test_case(module_factory=lambda: LinspaceTwoSizeModule())
def LinspaceTwoSizeModule_basic(module, tu: TestUtils):
    module.forward()


class PrimsIotaModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        return torch.ops.prims.iota(
            77, start=0, step=1, dtype=torch.int64, device="cpu", requires_grad=False
        )


@register_test_case(module_factory=lambda: PrimsIotaModule())
def PrimsIotaModule_basic(module, tu: TestUtils):
    module.forward()
