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


class AtenStftCenter2D(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([3, 35], torch.float32, True),
            ([4], torch.float32, True),
        ]
    )
    def forward(self, input, window):
        return torch.stft(
            input,
            n_fft=5,
            hop_length=2,
            win_length=4,
            window=window,
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )


@register_test_case(module_factory=lambda: AtenStftCenter2D())
def AtenStftCenter2D_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 35), tu.rand(4))
