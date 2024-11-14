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


class AtenStftCenter1D(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([40], torch.float32, True),
            ([4], torch.float32, True),
        ]
    )
    def forward(self, input, window):
        return input.stft(
            n_fft=4,
            hop_length=1,
            win_length=4,
            window=window,
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )


@register_test_case(module_factory=lambda: AtenStftCenter1D())
def AtenStftCenter1D_basic(module, tu: TestUtils):
    module.forward(tu.rand(40), tu.rand(4))


# ==============================================================================


class AtenStftCenter2D(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([4, 46], torch.float32, True),
            ([7], torch.float32, True),
        ]
    )
    def forward(self, input, window):
        return input.stft(
            n_fft=7,
            hop_length=1,
            win_length=7,
            window=window,
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )


@register_test_case(module_factory=lambda: AtenStftCenter2D())
def AtenStftCenter2D_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 46), tu.rand(7))


# ==============================================================================


class AtenStftCenter2DHopLength2(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([2, 61], torch.float32, True),
            ([8], torch.float32, True),
        ]
    )
    def forward(self, input, window):
        return input.stft(
            n_fft=8,
            hop_length=2,
            win_length=8,
            window=window,
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )


@register_test_case(module_factory=lambda: AtenStftCenter2DHopLength2())
def AtenStftCenter2DHopLength2_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 61), tu.rand(8))


# ==============================================================================


class AtenStftCenter2DWindowPadLeft(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([2, 68], torch.float32, True),
            ([6], torch.float32, True),
        ]
    )
    def forward(self, input, window):
        return input.stft(
            n_fft=7,
            hop_length=2,
            win_length=6,
            window=window,
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )


@register_test_case(module_factory=lambda: AtenStftCenter2DWindowPadLeft())
def AtenStftCenter2DWindowPadLeft_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 68), tu.rand(6))


# ==============================================================================


class AtenStftCenter2DHopLength3WindowPadBoth(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([3, 90], torch.float32, True),
            ([8], torch.float32, True),
        ]
    )
    def forward(self, input, window):
        return input.stft(
            n_fft=10,
            hop_length=3,
            win_length=8,
            window=window,
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )


@register_test_case(module_factory=lambda: AtenStftCenter2DHopLength3WindowPadBoth())
def AtenStftCenter2DHopLength3WindowPadBoth_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 90), tu.rand(8))
