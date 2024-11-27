# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================


class AtenHannWindowPeriodicFalseModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        return torch.ops.aten.hann_window(20, False)


@register_test_case(module_factory=lambda: AtenHannWindowPeriodicFalseModule())
def AtenHannWindowPeriodicFalseModule_basic(module, tu: TestUtils):
    module.forward()


# ==============================================================================


class AtenHannWindowPeriodicTrueModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        return torch.ops.aten.hann_window(20, True)


@register_test_case(module_factory=lambda: AtenHannWindowPeriodicTrueModule())
def AtenHannWindowPeriodicTrueModule_basic(module, tu: TestUtils):
    module.forward()


# ==============================================================================


class AtenFftRfft2DLastDim(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([16, 9], torch.float32, True),
        ]
    )
    def forward(self, input):
        return torch.fft.rfft(input, dim=-1)


@register_test_case(module_factory=lambda: AtenFftRfft2DLastDim())
def AtenFftRfft2DLastDim_basic(module, tu: TestUtils):
    module.forward(tu.rand(16, 9))


# ==============================================================================


class AtenFftRfft2DMiddleDim(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([36, 10], torch.float32, True),
        ]
    )
    def forward(self, input):
        return torch.fft.rfft(input, dim=0)


@register_test_case(module_factory=lambda: AtenFftRfft2DMiddleDim())
def AtenFftRfft2DMiddleDim_basic(module, tu: TestUtils):
    module.forward(tu.rand(36, 10))
