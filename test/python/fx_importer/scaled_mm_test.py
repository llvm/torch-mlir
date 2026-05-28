# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

import torch
import torch.nn as nn

from torch_mlir import fx


def make_fp4_tensor(shape, stride=None):
    if stride is None:
        tensor = torch.empty(shape, dtype=torch.float4_e2m1fn_x2)
    else:
        tensor = torch.empty_strided(shape, stride, dtype=torch.float4_e2m1fn_x2)
    # PyTorch's FP4 shell dtype does not implement fill_ directly.
    tensor.view(torch.uint8).fill_(1)
    return tensor


def run(f):
    print(f"{f.__name__}")
    print("-" * len(f.__name__))
    f()
    print()


@run
# CHECK-LABEL: test_import_scaled_mm_per_tensor
# CHECK: func.func @test_import_scaled_mm_per_tensor(%arg0: !torch.vtensor<[128,128],f8E4M3FN>, %arg1: !torch.vtensor<[128,128],f8E4M3FN>, %arg2: !torch.vtensor<[],f32>, %arg3: !torch.vtensor<[],f32>) -> !torch.vtensor<[128,128],bf16>
# CHECK: %[[NONE:.+]] = torch.constant.none
# CHECK: %[[NONE_0:.+]] = torch.constant.none
# CHECK: %[[INT15:.+]] = torch.constant.int 15
# CHECK: %[[FALSE:.+]] = torch.constant.bool false
# CHECK: %[[MM:.+]] = torch.aten._scaled_mm %arg0, %arg1, %arg2, %arg3, %[[NONE]], %[[NONE_0]], %[[INT15]], %[[FALSE]]
# CHECK: return %[[MM]]
def test_import_scaled_mm_per_tensor():
    class Basic(nn.Module):
        def forward(self, a, b, a_scale, b_scale):
            return torch._scaled_mm(a, b, a_scale, b_scale, out_dtype=torch.bfloat16)

    a = torch.ones((128, 128), dtype=torch.float32).to(torch.float8_e4m3fn)
    b = torch.ones((128, 128), dtype=torch.float32).to(torch.float8_e4m3fn)
    a_scale = torch.tensor(1.0, dtype=torch.float32)
    b_scale = torch.tensor(1.0, dtype=torch.float32)

    m = fx.export_and_import(
        Basic(),
        a,
        b,
        a_scale,
        b_scale,
        func_name="test_import_scaled_mm_per_tensor",
    )
    print(m)


@run
# CHECK-LABEL: test_import_scaled_mm_block_scaled_fp4
# CHECK: func.func @test_import_scaled_mm_block_scaled_fp4(%arg0: !torch.vtensor<[128,64],f4E2M1FN>, %arg1: !torch.vtensor<[64,128],f4E2M1FN>, %arg2: !torch.vtensor<[512],f8E8M0FNU>, %arg3: !torch.vtensor<[512],f8E8M0FNU>) -> !torch.vtensor<[128,128],bf16>
# CHECK: %[[NONE:.+]] = torch.constant.none
# CHECK: %[[NONE_0:.+]] = torch.constant.none
# CHECK: %[[INT15:.+]] = torch.constant.int 15
# CHECK: %[[FALSE:.+]] = torch.constant.bool false
# CHECK: %[[MM:.+]] = torch.aten._scaled_mm %arg0, %arg1, %arg2, %arg3, %[[NONE]], %[[NONE_0]], %[[INT15]], %[[FALSE]]
# CHECK: return %[[MM]]
def test_import_scaled_mm_block_scaled_fp4():
    class Basic(nn.Module):
        def forward(self, a, b, a_scale_block, b_scale_block):
            return torch._scaled_mm(
                a, b, a_scale_block, b_scale_block, out_dtype=torch.bfloat16
            )

    a = make_fp4_tensor((128, 64))
    b = make_fp4_tensor((64, 128), stride=(1, 64))
    a_scale_block = torch.zeros((512,), dtype=torch.float8_e8m0fnu)
    b_scale_block = torch.zeros((512,), dtype=torch.float8_e8m0fnu)

    m = fx.export_and_import(
        Basic(),
        a,
        b,
        a_scale_block,
        b_scale_block,
        func_name="test_import_scaled_mm_block_scaled_fp4",
    )
    print(m)


@run
# CHECK-LABEL: test_import_scaled_mm_per_tensor_e5m2
# CHECK: func.func @test_import_scaled_mm_per_tensor_e5m2(%arg0: !torch.vtensor<[128,128],f8E5M2>, %arg1: !torch.vtensor<[128,128],f8E5M2>, %arg2: !torch.vtensor<[],f32>, %arg3: !torch.vtensor<[],f32>) -> !torch.vtensor<[128,128],bf16>
# CHECK: %[[MM:.+]] = torch.aten._scaled_mm %arg0, %arg1, %arg2, %arg3
def test_import_scaled_mm_per_tensor_e5m2():
    class Basic(nn.Module):
        def forward(self, a, b, a_scale, b_scale):
            return torch._scaled_mm(a, b, a_scale, b_scale, out_dtype=torch.bfloat16)

    a = torch.ones((128, 128), dtype=torch.float32).to(torch.float8_e5m2)
    b = torch.ones((128, 128), dtype=torch.float32).to(torch.float8_e5m2)
    a_scale = torch.tensor(1.0, dtype=torch.float32)
    b_scale = torch.tensor(1.0, dtype=torch.float32)

    m = fx.export_and_import(
        Basic(),
        a,
        b,
        a_scale,
        b_scale,
        func_name="test_import_scaled_mm_per_tensor_e5m2",
    )
    print(m)


@run
# CHECK-LABEL: test_import_scaled_mm_out_dtype_none
# CHECK: func.func @test_import_scaled_mm_out_dtype_none(%arg0: !torch.vtensor<[128,128],f8E4M3FN>, %arg1: !torch.vtensor<[128,128],f8E5M2>, %arg2: !torch.vtensor<[],f32>, %arg3: !torch.vtensor<[],f32>) -> !torch.vtensor<[128,128],f8E4M3FN>
# CHECK: %[[MM:.+]] = torch.aten._scaled_mm %arg0, %arg1, %arg2, %arg3
# CHECK: return %[[MM]]
def test_import_scaled_mm_out_dtype_none():
    class Basic(nn.Module):
        def forward(self, a, b, a_scale, b_scale):
            return torch._scaled_mm(a, b, a_scale, b_scale, out_dtype=None)

    a = torch.ones((128, 128), dtype=torch.float32).to(torch.float8_e4m3fn)
    b = torch.ones((128, 128), dtype=torch.float32).to(torch.float8_e5m2)
    a_scale = torch.tensor(1.0, dtype=torch.float32)
    b_scale = torch.tensor(1.0, dtype=torch.float32)

    m = fx.export_and_import(
        Basic(),
        a,
        b,
        a_scale,
        b_scale,
        func_name="test_import_scaled_mm_out_dtype_none",
    )
    print(m)


@run
# CHECK-LABEL: test_import_scaled_mm_block_scaled_fp8
# CHECK: func.func @test_import_scaled_mm_block_scaled_fp8(%arg0: !torch.vtensor<[128,128],f8E4M3FN>, %arg1: !torch.vtensor<[128,128],f8E4M3FN>, %arg2: !torch.vtensor<[512],f8E8M0FNU>, %arg3: !torch.vtensor<[512],f8E8M0FNU>) -> !torch.vtensor<[128,128],bf16>
# CHECK: %[[NONE:.+]] = torch.constant.none
# CHECK: %[[NONE_0:.+]] = torch.constant.none
# CHECK: %[[INT15:.+]] = torch.constant.int 15
# CHECK: %[[FALSE:.+]] = torch.constant.bool false
# CHECK: %[[MM:.+]] = torch.aten._scaled_mm %arg0, %arg1, %arg2, %arg3, %[[NONE]], %[[NONE_0]], %[[INT15]], %[[FALSE]]
# CHECK: return %[[MM]]
def test_import_scaled_mm_block_scaled_fp8():
    class Basic(nn.Module):
        def forward(self, a, b, a_scale_block, b_scale_block):
            return torch._scaled_mm(
                a, b, a_scale_block, b_scale_block, out_dtype=torch.bfloat16
            )

    a = torch.ones((128, 128), dtype=torch.float32).to(torch.float8_e4m3fn)
    b = torch.ones((128, 128), dtype=torch.float32).to(torch.float8_e4m3fn)
    a_scale_block = torch.zeros((512,), dtype=torch.float8_e8m0fnu)
    b_scale_block = torch.zeros((512,), dtype=torch.float8_e8m0fnu)

    m = fx.export_and_import(
        Basic(),
        a,
        b,
        a_scale_block,
        b_scale_block,
        func_name="test_import_scaled_mm_block_scaled_fp8",
    )
    print(m)


@run
# CHECK-LABEL: test_import_scaled_mm_block_scaled_fp8_e5m2
# CHECK: func.func @test_import_scaled_mm_block_scaled_fp8_e5m2(%arg0: !torch.vtensor<[128,128],f8E5M2>, %arg1: !torch.vtensor<[128,128],f8E5M2>, %arg2: !torch.vtensor<[512],f8E8M0FNU>, %arg3: !torch.vtensor<[512],f8E8M0FNU>) -> !torch.vtensor<[128,128],bf16>
# CHECK: %[[MM:.+]] = torch.aten._scaled_mm %arg0, %arg1, %arg2, %arg3
def test_import_scaled_mm_block_scaled_fp8_e5m2():
    class Basic(nn.Module):
        def forward(self, a, b, a_scale_block, b_scale_block):
            return torch._scaled_mm(
                a, b, a_scale_block, b_scale_block, out_dtype=torch.bfloat16
            )

    a = torch.ones((128, 128), dtype=torch.float32).to(torch.float8_e5m2)
    b = torch.ones((128, 128), dtype=torch.float32).to(torch.float8_e5m2)
    a_scale_block = torch.zeros((512,), dtype=torch.float8_e8m0fnu)
    b_scale_block = torch.zeros((512,), dtype=torch.float8_e8m0fnu)

    m = fx.export_and_import(
        Basic(),
        a,
        b,
        a_scale_block,
        b_scale_block,
        func_name="test_import_scaled_mm_block_scaled_fp8_e5m2",
    )
    print(m)
