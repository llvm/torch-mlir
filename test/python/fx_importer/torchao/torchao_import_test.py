# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

# Verifies that FX-imported graphs with torchao affine quantization ops
# produce first-class torch.torchao.* dialect ops (not torch.operator) when
# the backend lists them as legal, and are decomposed otherwise.

import torch
import torch.nn as nn

# Load the torchao op library (defines the custom op namespace).
import torchao  # noqa: F401

from torch_mlir import fx

LEGAL_OPS = ["torchao.quantize_affine", "torchao.dequantize_affine"]


def run(f):
    print(f"{f.__name__}")
    print("-" * len(f.__name__))
    f()
    print()


@run
# CHECK-LABEL: test_quantize_dequantize_affine
# CHECK: = torch.torchao.quantize_affine %
# CHECK: = torch.torchao.dequantize_affine %
# CHECK-NOT: torch.operator
def test_quantize_dequantize_affine():
    class QuantizeDequantize(nn.Module):
        def forward(self, x, scale, zero_point):
            q = torch.ops.torchao.quantize_affine.default(
                x, [1, 8], scale, zero_point, torch.int8, -128, 127
            )
            return torch.ops.torchao.dequantize_affine.default(
                q, [1, 8], scale, zero_point, torch.int8, -128, 127
            )

    m = fx.export_and_import(
        QuantizeDequantize(),
        torch.randn(4, 16),
        torch.rand(4, 2) + 0.1,
        torch.zeros(4, 2, dtype=torch.int32),
        func_name="test_quantize_dequantize_affine",
        backend_legal_ops=LEGAL_OPS,
    )
    print(m)


@run
# CHECK-LABEL: test_decomposed_by_default
# CHECK-NOT: torch.torchao
# CHECK: torch.aten.round
def test_decomposed_by_default():
    class PerTensor(nn.Module):
        def forward(self, x, scale, zero_point):
            return torch.ops.torchao.quantize_affine.default(
                x, list(x.shape), scale, zero_point, torch.int8
            )

    m = fx.export_and_import(
        PerTensor(),
        torch.randn(4, 16),
        torch.tensor([0.1]),
        torch.tensor([0], dtype=torch.int32),
        func_name="test_decomposed_by_default",
    )
    print(m)
