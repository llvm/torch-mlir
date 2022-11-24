# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================

class TensorToIntZeroRank(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([], torch.int64, True),
    ])
    def forward(self, x):
        return int(x)


@register_test_case(module_factory=lambda: TensorToIntZeroRank())
def TensorToIntZeroRank_basic(module, tu: TestUtils):
    module.forward(tu.randint(high=10))

# ==============================================================================

class TensorToInt(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, x):
        return int(x)


@register_test_case(module_factory=lambda: TensorToInt())
def TensorToInt_basic(module, tu: TestUtils):
    module.forward(tu.randint(1, 1, high=10))

# ==============================================================================

class TensorToFloatZeroRank(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([], torch.float64, True),
    ])
    def forward(self, x):
        return float(x)


@register_test_case(module_factory=lambda: TensorToFloatZeroRank())
def TensorToFloatZeroRank_basic(module, tu: TestUtils):
    module.forward(torch.rand((), dtype=torch.float64))

# ==============================================================================

class TensorToFloat(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float64, True),
    ])
    def forward(self, x):
        return float(x)


@register_test_case(module_factory=lambda: TensorToFloat())
def TensorToFloat_basic(module, tu: TestUtils):
    module.forward(torch.rand((1, 1), dtype=torch.float64))

# ==============================================================================

class TensorToBoolZeroRank(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([], torch.bool, True),
    ])
    def forward(self, x):
        return bool(x)


@register_test_case(module_factory=lambda: TensorToBoolZeroRank())
def TensorToBoolZeroRank_basic(module, tu: TestUtils):
    module.forward(torch.tensor(1, dtype=torch.bool))

# ==============================================================================

class TensorToBool(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.bool, True),
    ])
    def forward(self, x):
        return bool(x)


@register_test_case(module_factory=lambda: TensorToBool())
def TensorToBool_basic(module, tu: TestUtils):
    module.forward(torch.tensor([[1]], dtype=torch.bool))
