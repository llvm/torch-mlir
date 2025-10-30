# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from numpy import int64
import torch
import random

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export
from torch._higher_order_ops.while_loop import while_loop

# ==============================================================================


class TorchPrimLoopForLikeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1, -1], torch.int64, True)])
    def forward(self, x):
        x_val = x.size(0)
        sum = 0
        for i in range(x_val):
            sum += i
        return sum


@register_test_case(module_factory=lambda: TorchPrimLoopForLikeModule())
def TorchPrimLoopForLikeModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(6, 8, high=10))


# ==============================================================================
class TorchPrimLoopWhileLikeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1, -1], torch.int64, True)])
    def forward(self, x):
        x_val = x.size(0)
        sum = 0
        while x_val > sum:
            sum += 1
        return sum


@register_test_case(module_factory=lambda: TorchPrimLoopWhileLikeModule())
def TorchPrimLoopWhileLikeModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(6, 8, high=10))


# ==============================================================================


class TorchPrimLoopForLikeTensorArgModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([7, 9], torch.float32, True),
        ]
    )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(50):
            x = x + i
        return x


@register_test_case(module_factory=lambda: TorchPrimLoopForLikeTensorArgModule())
def TorchPrimLoopForLikeTensorArgModule_basic(module, tu: TestUtils):
    x_test = torch.zeros([7, 9]).float()

    module.forward(x_test)


# ==============================================================================


class TorchPrimLoopWhileLikeHOPModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def body_fn(self, i, x):
        return i + 1, x + 1

    def cond_fn(self, i, x):
        return i < 3

    @export
    @annotate_args(
        [
            None,
            ([7, 9], torch.float32, True),
        ]
    )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        i0 = torch.tensor(0)
        out_i, out_x = while_loop(self.cond_fn, self.body_fn, (i0, x))
        return out_i, out_x


@register_test_case(module_factory=lambda: TorchPrimLoopWhileLikeHOPModule())
def TorchPrimLoopWhileLikeHOPModule_basic(module, tu: TestUtils):
    x_test = torch.zeros([7, 9]).float()

    module.forward(x_test)
