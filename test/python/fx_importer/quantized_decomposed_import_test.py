# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

# Verifies that FX-imported PT2E-style graphs with quantized_decomposed ops
# produce first-class torch.quantized_decomposed.* dialect ops (not torch.operator).

import torch
import torch.nn as nn

# Load the quantized_decomposed op library (defines the custom op namespace).
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401

from torch_mlir import fx


def run(f):
    print(f"{f.__name__}")
    print("-" * len(f.__name__))
    f()
    print()


@run
# CHECK-LABEL: test_quantize_per_tensor
# CHECK: func.func @test_quantize_per_tensor
# CHECK: %[[SCALE:.+]] = torch.constant.float
# CHECK: %[[ZP:.+]] = torch.constant.int
# CHECK: torch.quantized_decomposed.quantize_per_tensor
# CHECK-NOT: torch.operator
def test_quantize_per_tensor():
    class QuantizeOnly(nn.Module):
        def forward(self, x):
            return torch.ops.quantized_decomposed.quantize_per_tensor.default(
                x, 0.3, 0, -128, 127, torch.int8
            )

    m = fx.export_and_import(
        QuantizeOnly(),
        torch.randn(4, 8),
        func_name="test_quantize_per_tensor",
    )
    print(m)


@run
# CHECK-LABEL: test_dequantize_per_tensor
# CHECK: func.func @test_dequantize_per_tensor
# CHECK: torch.quantized_decomposed.dequantize_per_tensor
# CHECK-NOT: torch.operator
def test_dequantize_per_tensor():
    class DequantizeOnly(nn.Module):
        def forward(self, x):
            return torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                x, 0.3, 0, -128, 127, torch.int8
            )

    m = fx.export_and_import(
        DequantizeOnly(),
        torch.zeros(4, 8, dtype=torch.int8),
        func_name="test_dequantize_per_tensor",
    )
    print(m)


@run
# CHECK-LABEL: test_dq_mm_q_chain
# CHECK: func.func @test_dq_mm_q_chain
# CHECK: torch.quantized_decomposed.dequantize_per_tensor
# CHECK: torch.quantized_decomposed.dequantize_per_tensor
# CHECK: torch.aten.mm
# CHECK: torch.quantized_decomposed.quantize_per_tensor
# CHECK-NOT: torch.operator
def test_dq_mm_q_chain():
    class DqMmQ(nn.Module):
        def forward(self, lhs_q, rhs_q):
            lhs_f = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                lhs_q, 0.3, 0, -128, 127, torch.int8
            )
            rhs_f = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                rhs_q, 0.2, 0, -128, 127, torch.int8
            )
            result_f = torch.mm(lhs_f, rhs_f)
            return torch.ops.quantized_decomposed.quantize_per_tensor.default(
                result_f, 0.5, 0, -128, 127, torch.int8
            )

    m = fx.export_and_import(
        DqMmQ(),
        torch.zeros(4, 8, dtype=torch.int8),
        torch.zeros(8, 16, dtype=torch.int8),
        func_name="test_dq_mm_q_chain",
    )
    print(m)
