# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch
import torch.nn.functional as F

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================

# torch.nn.functional.unfold calls aten::im2col, for both batched 4-D inputs and
# unbatched 3-D ones.


class Im2colModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1, 2, 4, 4], torch.float32, True),
        ]
    )
    def forward(self, x):
        return F.unfold(x, kernel_size=(2, 2))


@register_test_case(module_factory=lambda: Im2colModule())
def Im2colModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 2, 4, 4))


# ==============================================================================


class Im2colPaddingStrideModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 3, 5, 5], torch.float32, True),
        ]
    )
    def forward(self, x):
        return F.unfold(x, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))


@register_test_case(module_factory=lambda: Im2colPaddingStrideModule())
def Im2colPaddingStrideModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 5, 5))


# ==============================================================================


class Im2colDilationModule(torch.nn.Module):
    """Dilation on one axis only, with a non-square kernel."""

    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1, 1, 6, 6], torch.float32, True),
        ]
    )
    def forward(self, x):
        return F.unfold(x, kernel_size=(2, 3), dilation=(2, 1))


@register_test_case(module_factory=lambda: Im2colDilationModule())
def Im2colDilationModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 6, 6))


# ==============================================================================


class Im2colAllParamsModule(torch.nn.Module):
    """Dilation, padding and stride all non-trivial and asymmetric."""

    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1, 2, 7, 5], torch.float32, True),
        ]
    )
    def forward(self, x):
        return F.unfold(
            x, kernel_size=(2, 2), dilation=(2, 2), padding=(1, 2), stride=(3, 1)
        )


@register_test_case(module_factory=lambda: Im2colAllParamsModule())
def Im2colAllParamsModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 2, 7, 5))


# ==============================================================================


class Im2colUnbatchedModule(torch.nn.Module):
    """A 3-D input; aten::im2col takes it unbatched and returns a 2-D result."""

    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 4, 4], torch.float32, True),
        ]
    )
    def forward(self, x):
        return F.unfold(x, kernel_size=(2, 2))


@register_test_case(module_factory=lambda: Im2colUnbatchedModule())
def Im2colUnbatchedModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4, 4))
