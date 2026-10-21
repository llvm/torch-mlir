# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s torch-mlir-opt | FileCheck %s

import subprocess
import sys

import numpy as np

from torch_mlir import ir
from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import (
    RefBackendLinalgOnTensorsBackend,
)


# Exercise both sides of 128, unsigned extremes, and nonzero zero points.
# Use two distinct batches so a batch indexing error cannot pass unnoticed.
lhs = np.array(
    [[[0, 127, 128], [137, 200, 255]], [[255, 1, 149], [128, 254, 3]]],
    dtype=np.uint8,
)
rhs = np.array(
    [[[255, 149], [128, 0], [127, 200]], [[1, 255], [149, 128], [250, 127]]],
    dtype=np.uint8,
)
expected = (lhs.astype(np.int32) - 137) @ (rhs.astype(np.int32) - 149)

# Use Torch IR directly to exercise quantized bmm without frontend fusion.
TORCH_IR = """
module {
  func.func @main(%lhs: !torch.vtensor<[2,2,3],ui8>, %rhs: !torch.vtensor<[2,3,2],ui8>) -> !torch.vtensor<[2,2,2],si32> {
    %scale = torch.constant.float 1.000000e+00
    %lhs_zp = torch.constant.int 137
    %rhs_zp = torch.constant.int 149
    %lhs_q = torch.aten._make_per_tensor_quantized_tensor %lhs, %scale, %lhs_zp : !torch.vtensor<[2,2,3],ui8>, !torch.float, !torch.int -> !torch.vtensor<[2,2,3],!torch.quint8>
    %rhs_q = torch.aten._make_per_tensor_quantized_tensor %rhs, %scale, %rhs_zp : !torch.vtensor<[2,3,2],ui8>, !torch.float, !torch.int -> !torch.vtensor<[2,3,2],!torch.quint8>
    %result = torch.aten.bmm %lhs_q, %rhs_q : !torch.vtensor<[2,2,3],!torch.quint8>, !torch.vtensor<[2,3,2],!torch.quint8> -> !torch.vtensor<[2,2,2],si32>
    return %result : !torch.vtensor<[2,2,2],si32>
  }
}
"""

# Lower with the executable under test, including its unsigned sign shifts.
lowered = subprocess.check_output(
    [
        sys.argv[1],
        "-torch-backend-to-linalg-on-tensors-backend-pipeline",
    ],
    input=TORCH_IR,
    text=True,
)
assert "linalg.quantized_batch_matmul" in lowered
with ir.Context():
    module = ir.Module.parse(lowered)
    backend = RefBackendLinalgOnTensorsBackend()
    invoker = backend.load(backend.compile(module))
    actual = invoker.main(lhs, rhs)
    np.testing.assert_array_equal(actual, expected)

# CHECK: PASS: unsigned quantized bmm
print("PASS: unsigned quantized bmm")
