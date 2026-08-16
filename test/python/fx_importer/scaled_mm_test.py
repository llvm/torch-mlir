# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

import torch
import torch.nn as nn
import torch.nn.functional as F

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
# CHECK-LABEL: test_import_scaled_mm_v2_block_scaled_fp4
# CHECK: func.func @test_import_scaled_mm_v2_block_scaled_fp4(%arg0: !torch.vtensor<[128,64],f4E2M1FN>, %arg1: !torch.vtensor<[64,128],f4E2M1FN>, %arg2: !torch.vtensor<[512],f8E8M0FNU>, %arg3: !torch.vtensor<[512],f8E8M0FNU>) -> !torch.vtensor<[128,128],bf16>
# CHECK-NOT: torch.operator
# CHECK: %[[SCALE_A:.*]] = torch.prim.ListConstruct %arg2
# CHECK: %[[RECIPE_A_VALUE:.*]] = torch.constant.int 3
# CHECK: %[[RECIPE_A:.*]] = torch.prim.ListConstruct %[[RECIPE_A_VALUE]]
# CHECK: %[[SWIZZLE_A_VALUE:.*]] = torch.constant.int 1
# CHECK: %[[SWIZZLE_A:.*]] = torch.prim.ListConstruct %[[SWIZZLE_A_VALUE]]
# CHECK: %[[SCALE_B:.*]] = torch.prim.ListConstruct %arg3
# CHECK: %[[RECIPE_B_VALUE:.*]] = torch.constant.int 3
# CHECK: %[[RECIPE_B:.*]] = torch.prim.ListConstruct %[[RECIPE_B_VALUE]]
# CHECK: %[[SWIZZLE_B_VALUE:.*]] = torch.constant.int 1
# CHECK: %[[SWIZZLE_B:.*]] = torch.prim.ListConstruct %[[SWIZZLE_B_VALUE]]
# CHECK: %[[NONE:.*]] = torch.constant.none
# CHECK: %[[OUT_DTYPE:.*]] = torch.constant.int 15
# CHECK: %[[CONTRACTION:.*]] = torch.prim.ListConstruct
# CHECK: %[[FALSE:.*]] = torch.constant.bool false
# CHECK: %[[MM:.*]] = torch.aten._scaled_mm_v2 %arg0, %arg1, %[[SCALE_A]], %[[RECIPE_A]], %[[SWIZZLE_A]], %[[SCALE_B]], %[[RECIPE_B]], %[[SWIZZLE_B]], %[[NONE]], %[[OUT_DTYPE]], %[[CONTRACTION]], %[[FALSE]]
# CHECK: return %[[MM]]
def test_import_scaled_mm_v2_block_scaled_fp4():
    class Basic(nn.Module):
        def forward(self, a, b, a_scale_block, b_scale_block):
            return F.scaled_mm(
                a,
                b,
                scale_a=[a_scale_block],
                scale_recipe_a=[F.ScalingType.BlockWise1x32],
                swizzle_a=[F.SwizzleType.SWIZZLE_32_4_4],
                scale_b=[b_scale_block],
                scale_recipe_b=[F.ScalingType.BlockWise1x32],
                swizzle_b=[F.SwizzleType.SWIZZLE_32_4_4],
                bias=None,
                output_dtype=torch.bfloat16,
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
        func_name="test_import_scaled_mm_v2_block_scaled_fp4",
    )
    print(m)


@run
# CHECK-LABEL: test_import_scaled_mm_v2_per_tensor_fp4
# CHECK: func.func @test_import_scaled_mm_v2_per_tensor_fp4(%arg0: !torch.vtensor<[128,64],f4E2M1FN>, %arg1: !torch.vtensor<[64,128],f4E2M1FN>, %arg2: !torch.vtensor<[],f32>, %arg3: !torch.vtensor<[],f32>) -> !torch.vtensor<[128,128],bf16>
# CHECK-NOT: torch.operator
# CHECK: %[[SCALE_A:.*]] = torch.prim.ListConstruct %arg2
# CHECK: %[[RECIPE_A_VALUE:.*]] = torch.constant.int 0
# CHECK: %[[RECIPE_A:.*]] = torch.prim.ListConstruct %[[RECIPE_A_VALUE]]
# CHECK: %[[SWIZZLE_A_VALUE:.*]] = torch.constant.int 0
# CHECK: %[[SWIZZLE_A:.*]] = torch.prim.ListConstruct %[[SWIZZLE_A_VALUE]]
# CHECK: %[[SCALE_B:.*]] = torch.prim.ListConstruct %arg3
# CHECK: %[[RECIPE_B_VALUE:.*]] = torch.constant.int 0
# CHECK: %[[RECIPE_B:.*]] = torch.prim.ListConstruct %[[RECIPE_B_VALUE]]
# CHECK: %[[SWIZZLE_B_VALUE:.*]] = torch.constant.int 0
# CHECK: %[[SWIZZLE_B:.*]] = torch.prim.ListConstruct %[[SWIZZLE_B_VALUE]]
# CHECK: %[[NONE:.*]] = torch.constant.none
# CHECK: %[[OUT_DTYPE:.*]] = torch.constant.int 15
# CHECK: %[[CONTRACTION:.*]] = torch.prim.ListConstruct
# CHECK: %[[FALSE:.*]] = torch.constant.bool false
# CHECK: %[[MM:.*]] = torch.aten._scaled_mm_v2 %arg0, %arg1, %[[SCALE_A]], %[[RECIPE_A]], %[[SWIZZLE_A]], %[[SCALE_B]], %[[RECIPE_B]], %[[SWIZZLE_B]], %[[NONE]], %[[OUT_DTYPE]], %[[CONTRACTION]], %[[FALSE]]
# CHECK: return %[[MM]]
def test_import_scaled_mm_v2_per_tensor_fp4():
    class Basic(nn.Module):
        def forward(self, a, b, a_scale, b_scale):
            return F.scaled_mm(
                a,
                b,
                scale_a=[a_scale],
                scale_recipe_a=[F.ScalingType.TensorWise],
                swizzle_a=[F.SwizzleType.NO_SWIZZLE],
                scale_b=[b_scale],
                scale_recipe_b=[F.ScalingType.TensorWise],
                swizzle_b=[F.SwizzleType.NO_SWIZZLE],
                bias=None,
                output_dtype=torch.bfloat16,
            )

    a = make_fp4_tensor((128, 64))
    b = make_fp4_tensor((64, 128), stride=(1, 64))
    a_scale = torch.tensor(1.0, dtype=torch.float32)
    b_scale = torch.tensor(1.0, dtype=torch.float32)

    m = fx.export_and_import(
        Basic(),
        a,
        b,
        a_scale,
        b_scale,
        func_name="test_import_scaled_mm_v2_per_tensor_fp4",
    )
    print(m)


@run
# CHECK-LABEL: test_import_scaled_mm_v2_out_dtype_none_fp4
# CHECK: func.func @test_import_scaled_mm_v2_out_dtype_none_fp4(%arg0: !torch.vtensor<[128,64],f4E2M1FN>, %arg1: !torch.vtensor<[64,128],f4E2M1FN>, %arg2: !torch.vtensor<[],f32>, %arg3: !torch.vtensor<[],f32>) -> !torch.vtensor<[128,128],f4E2M1FN>
# CHECK-NOT: torch.operator
# CHECK: %[[SCALE_A:.*]] = torch.prim.ListConstruct %arg2
# CHECK: %[[RECIPE_A_VALUE:.*]] = torch.constant.int 0
# CHECK: %[[RECIPE_A:.*]] = torch.prim.ListConstruct %[[RECIPE_A_VALUE]]
# CHECK: %[[SWIZZLE_A_VALUE:.*]] = torch.constant.int 0
# CHECK: %[[SWIZZLE_A:.*]] = torch.prim.ListConstruct %[[SWIZZLE_A_VALUE]]
# CHECK: %[[SCALE_B:.*]] = torch.prim.ListConstruct %arg3
# CHECK: %[[RECIPE_B_VALUE:.*]] = torch.constant.int 0
# CHECK: %[[RECIPE_B:.*]] = torch.prim.ListConstruct %[[RECIPE_B_VALUE]]
# CHECK: %[[SWIZZLE_B_VALUE:.*]] = torch.constant.int 0
# CHECK: %[[SWIZZLE_B:.*]] = torch.prim.ListConstruct %[[SWIZZLE_B_VALUE]]
# CHECK: %[[BIAS_NONE:.*]] = torch.constant.none
# CHECK: %[[OUT_DTYPE_NONE:.*]] = torch.constant.none
# CHECK: %[[CONTRACTION:.*]] = torch.prim.ListConstruct
# CHECK: %[[FALSE:.*]] = torch.constant.bool false
# CHECK: %[[MM:.*]] = torch.aten._scaled_mm_v2 %arg0, %arg1, %[[SCALE_A]], %[[RECIPE_A]], %[[SWIZZLE_A]], %[[SCALE_B]], %[[RECIPE_B]], %[[SWIZZLE_B]], %[[BIAS_NONE]], %[[OUT_DTYPE_NONE]], %[[CONTRACTION]], %[[FALSE]]
# CHECK: return %[[MM]]
def test_import_scaled_mm_v2_out_dtype_none_fp4():
    class Basic(nn.Module):
        def forward(self, a, b, a_scale, b_scale):
            return F.scaled_mm(
                a,
                b,
                scale_a=[a_scale],
                scale_recipe_a=[F.ScalingType.TensorWise],
                swizzle_a=[F.SwizzleType.NO_SWIZZLE],
                scale_b=[b_scale],
                scale_recipe_b=[F.ScalingType.TensorWise],
                swizzle_b=[F.SwizzleType.NO_SWIZZLE],
                bias=None,
                output_dtype=None,
            )

    a = make_fp4_tensor((128, 64))
    b = make_fp4_tensor((64, 128), stride=(1, 64))
    a_scale = torch.tensor(1.0, dtype=torch.float32)
    b_scale = torch.tensor(1.0, dtype=torch.float32)

    m = fx.export_and_import(
        Basic(),
        a,
        b,
        a_scale,
        b_scale,
        func_name="test_import_scaled_mm_v2_out_dtype_none_fp4",
    )
    print(m)


@run
# CHECK-LABEL: test_import_scaled_mm_v2_out_dtype_fp4
# CHECK: func.func @test_import_scaled_mm_v2_out_dtype_fp4(%arg0: !torch.vtensor<[128,64],f4E2M1FN>, %arg1: !torch.vtensor<[64,128],f4E2M1FN>, %arg2: !torch.vtensor<[],f32>, %arg3: !torch.vtensor<[],f32>) -> !torch.vtensor<[128,128],f4E2M1FN>
# CHECK-NOT: torch.operator
# CHECK: %[[SCALE_A:.*]] = torch.prim.ListConstruct %arg2
# CHECK: %[[RECIPE_A_VALUE:.*]] = torch.constant.int 0
# CHECK: %[[RECIPE_A:.*]] = torch.prim.ListConstruct %[[RECIPE_A_VALUE]]
# CHECK: %[[SWIZZLE_A_VALUE:.*]] = torch.constant.int 0
# CHECK: %[[SWIZZLE_A:.*]] = torch.prim.ListConstruct %[[SWIZZLE_A_VALUE]]
# CHECK: %[[SCALE_B:.*]] = torch.prim.ListConstruct %arg3
# CHECK: %[[RECIPE_B_VALUE:.*]] = torch.constant.int 0
# CHECK: %[[RECIPE_B:.*]] = torch.prim.ListConstruct %[[RECIPE_B_VALUE]]
# CHECK: %[[SWIZZLE_B_VALUE:.*]] = torch.constant.int 0
# CHECK: %[[SWIZZLE_B:.*]] = torch.prim.ListConstruct %[[SWIZZLE_B_VALUE]]
# CHECK: %[[BIAS_NONE:.*]] = torch.constant.none
# CHECK: %[[OUT_DTYPE:.*]] = torch.constant.int 29
# CHECK: %[[CONTRACTION:.*]] = torch.prim.ListConstruct
# CHECK: %[[FALSE:.*]] = torch.constant.bool false
# CHECK: %[[MM:.*]] = torch.aten._scaled_mm_v2 %arg0, %arg1, %[[SCALE_A]], %[[RECIPE_A]], %[[SWIZZLE_A]], %[[SCALE_B]], %[[RECIPE_B]], %[[SWIZZLE_B]], %[[BIAS_NONE]], %[[OUT_DTYPE]], %[[CONTRACTION]], %[[FALSE]]
# CHECK: return %[[MM]]
def test_import_scaled_mm_v2_out_dtype_fp4():
    class Basic(nn.Module):
        def forward(self, a, b, a_scale, b_scale):
            return F.scaled_mm(
                a,
                b,
                scale_a=[a_scale],
                scale_recipe_a=[F.ScalingType.TensorWise],
                swizzle_a=[F.SwizzleType.NO_SWIZZLE],
                scale_b=[b_scale],
                scale_recipe_b=[F.ScalingType.TensorWise],
                swizzle_b=[F.SwizzleType.NO_SWIZZLE],
                bias=None,
                output_dtype=torch.float4_e2m1fn_x2,
            )

    a = make_fp4_tensor((128, 64))
    b = make_fp4_tensor((64, 128), stride=(1, 64))
    a_scale = torch.tensor(1.0, dtype=torch.float32)
    b_scale = torch.tensor(1.0, dtype=torch.float32)

    m = fx.export_and_import(
        Basic(),
        a,
        b,
        a_scale,
        b_scale,
        func_name="test_import_scaled_mm_v2_out_dtype_fp4",
    )
    print(m)


@run
# CHECK-LABEL: test_import_scaled_mm_v2_rowwise_fp4
# CHECK: func.func @test_import_scaled_mm_v2_rowwise_fp4(%arg0: !torch.vtensor<[128,64],f4E2M1FN>, %arg1: !torch.vtensor<[64,128],f4E2M1FN>, %arg2: !torch.vtensor<[128,1],f32>, %arg3: !torch.vtensor<[128,1],f32>) -> !torch.vtensor<[128,128],bf16>
# CHECK-NOT: torch.operator
# CHECK: %[[SCALE_A:.*]] = torch.prim.ListConstruct %arg2
# CHECK: %[[RECIPE_A_VALUE:.*]] = torch.constant.int 1
# CHECK: %[[RECIPE_A:.*]] = torch.prim.ListConstruct %[[RECIPE_A_VALUE]]
# CHECK: %[[SWIZZLE_A_VALUE:.*]] = torch.constant.int 0
# CHECK: %[[SWIZZLE_A:.*]] = torch.prim.ListConstruct %[[SWIZZLE_A_VALUE]]
# CHECK: %[[SCALE_B:.*]] = torch.prim.ListConstruct %arg3
# CHECK: %[[RECIPE_B_VALUE:.*]] = torch.constant.int 1
# CHECK: %[[RECIPE_B:.*]] = torch.prim.ListConstruct %[[RECIPE_B_VALUE]]
# CHECK: %[[SWIZZLE_B_VALUE:.*]] = torch.constant.int 0
# CHECK: %[[SWIZZLE_B:.*]] = torch.prim.ListConstruct %[[SWIZZLE_B_VALUE]]
# CHECK: %[[NONE:.*]] = torch.constant.none
# CHECK: %[[OUT_DTYPE:.*]] = torch.constant.int 15
# CHECK: %[[CONTRACTION:.*]] = torch.prim.ListConstruct
# CHECK: %[[FALSE:.*]] = torch.constant.bool false
# CHECK: %[[MM:.*]] = torch.aten._scaled_mm_v2 %arg0, %arg1, %[[SCALE_A]], %[[RECIPE_A]], %[[SWIZZLE_A]], %[[SCALE_B]], %[[RECIPE_B]], %[[SWIZZLE_B]], %[[NONE]], %[[OUT_DTYPE]], %[[CONTRACTION]], %[[FALSE]]
# CHECK: return %[[MM]]
def test_import_scaled_mm_v2_rowwise_fp4():
    class Basic(nn.Module):
        def forward(self, a, b, a_scale, b_scale):
            return F.scaled_mm(
                a,
                b,
                scale_a=[a_scale],
                scale_recipe_a=[F.ScalingType.RowWise],
                swizzle_a=[F.SwizzleType.NO_SWIZZLE],
                scale_b=[b_scale],
                scale_recipe_b=[F.ScalingType.RowWise],
                swizzle_b=[F.SwizzleType.NO_SWIZZLE],
                bias=None,
                output_dtype=torch.bfloat16,
            )

    a = make_fp4_tensor((128, 64))
    b = make_fp4_tensor((64, 128), stride=(1, 64))
    a_scale = torch.zeros((128, 1), dtype=torch.float32)
    b_scale = torch.zeros((128, 1), dtype=torch.float32)

    m = fx.export_and_import(
        Basic(),
        a,
        b,
        a_scale,
        b_scale,
        func_name="test_import_scaled_mm_v2_rowwise_fp4",
    )
    print(m)


@run
# CHECK-LABEL: test_import_scaled_mm_v2_f32_blockwise_1x128_1x128_fp4
# CHECK: func.func @test_import_scaled_mm_v2_f32_blockwise_1x128_1x128_fp4(%arg0: !torch.vtensor<[128,64],f4E2M1FN>, %arg1: !torch.vtensor<[64,128],f4E2M1FN>, %arg2: !torch.vtensor<[128,1],f32>, %arg3: !torch.vtensor<[128,1],f32>) -> !torch.vtensor<[128,128],bf16>
# CHECK-NOT: torch.operator
# CHECK: %[[SCALE_A:.*]] = torch.prim.ListConstruct %arg2
# CHECK: %[[RECIPE_A_VALUE:.*]] = torch.constant.int 4
# CHECK: %[[RECIPE_A:.*]] = torch.prim.ListConstruct %[[RECIPE_A_VALUE]]
# CHECK: %[[SWIZZLE_A_VALUE:.*]] = torch.constant.int 0
# CHECK: %[[SWIZZLE_A:.*]] = torch.prim.ListConstruct %[[SWIZZLE_A_VALUE]]
# CHECK: %[[SCALE_B:.*]] = torch.prim.ListConstruct %arg3
# CHECK: %[[RECIPE_B_VALUE:.*]] = torch.constant.int 4
# CHECK: %[[RECIPE_B:.*]] = torch.prim.ListConstruct %[[RECIPE_B_VALUE]]
# CHECK: %[[SWIZZLE_B_VALUE:.*]] = torch.constant.int 0
# CHECK: %[[SWIZZLE_B:.*]] = torch.prim.ListConstruct %[[SWIZZLE_B_VALUE]]
# CHECK: %[[NONE:.*]] = torch.constant.none
# CHECK: %[[OUT_DTYPE:.*]] = torch.constant.int 15
# CHECK: %[[CONTRACTION:.*]] = torch.prim.ListConstruct
# CHECK: %[[FALSE:.*]] = torch.constant.bool false
# CHECK: %[[MM:.*]] = torch.aten._scaled_mm_v2 %arg0, %arg1, %[[SCALE_A]], %[[RECIPE_A]], %[[SWIZZLE_A]], %[[SCALE_B]], %[[RECIPE_B]], %[[SWIZZLE_B]], %[[NONE]], %[[OUT_DTYPE]], %[[CONTRACTION]], %[[FALSE]]
# CHECK: return %[[MM]]
def test_import_scaled_mm_v2_f32_blockwise_1x128_1x128_fp4():
    class Basic(nn.Module):
        def forward(self, a, b, a_scale, b_scale):
            return F.scaled_mm(
                a,
                b,
                scale_a=[a_scale],
                scale_recipe_a=[F.ScalingType.BlockWise1x128],
                swizzle_a=[F.SwizzleType.NO_SWIZZLE],
                scale_b=[b_scale],
                scale_recipe_b=[F.ScalingType.BlockWise1x128],
                swizzle_b=[F.SwizzleType.NO_SWIZZLE],
                bias=None,
                output_dtype=torch.bfloat16,
            )

    a = make_fp4_tensor((128, 64))
    b = make_fp4_tensor((64, 128), stride=(1, 64))
    a_scale = torch.zeros((128, 1), dtype=torch.float32)
    b_scale = torch.zeros((128, 1), dtype=torch.float32)

    m = fx.export_and_import(
        Basic(),
        a,
        b,
        a_scale,
        b_scale,
        func_name="test_import_scaled_mm_v2_f32_blockwise_1x128_1x128_fp4",
    )
    print(m)


@run
# CHECK-LABEL: test_import_scaled_mm_v2_f32_blockwise_1x128_128x128_fp4
# CHECK: func.func @test_import_scaled_mm_v2_f32_blockwise_1x128_128x128_fp4(%arg0: !torch.vtensor<[128,64],f4E2M1FN>, %arg1: !torch.vtensor<[64,128],f4E2M1FN>, %arg2: !torch.vtensor<[128,1],f32>, %arg3: !torch.vtensor<[4,1],f32>) -> !torch.vtensor<[128,128],bf16>
# CHECK-NOT: torch.operator
# CHECK: %[[SCALE_A:.*]] = torch.prim.ListConstruct %arg2
# CHECK: %[[RECIPE_A_VALUE:.*]] = torch.constant.int 4
# CHECK: %[[RECIPE_A:.*]] = torch.prim.ListConstruct %[[RECIPE_A_VALUE]]
# CHECK: %[[SWIZZLE_A_VALUE:.*]] = torch.constant.int 0
# CHECK: %[[SWIZZLE_A:.*]] = torch.prim.ListConstruct %[[SWIZZLE_A_VALUE]]
# CHECK: %[[SCALE_B:.*]] = torch.prim.ListConstruct %arg3
# CHECK: %[[RECIPE_B_VALUE:.*]] = torch.constant.int 5
# CHECK: %[[RECIPE_B:.*]] = torch.prim.ListConstruct %[[RECIPE_B_VALUE]]
# CHECK: %[[SWIZZLE_B_VALUE:.*]] = torch.constant.int 0
# CHECK: %[[SWIZZLE_B:.*]] = torch.prim.ListConstruct %[[SWIZZLE_B_VALUE]]
# CHECK: %[[NONE:.*]] = torch.constant.none
# CHECK: %[[OUT_DTYPE:.*]] = torch.constant.int 15
# CHECK: %[[CONTRACTION:.*]] = torch.prim.ListConstruct
# CHECK: %[[FALSE:.*]] = torch.constant.bool false
# CHECK: %[[MM:.*]] = torch.aten._scaled_mm_v2 %arg0, %arg1, %[[SCALE_A]], %[[RECIPE_A]], %[[SWIZZLE_A]], %[[SCALE_B]], %[[RECIPE_B]], %[[SWIZZLE_B]], %[[NONE]], %[[OUT_DTYPE]], %[[CONTRACTION]], %[[FALSE]]
# CHECK: return %[[MM]]
def test_import_scaled_mm_v2_f32_blockwise_1x128_128x128_fp4():
    class Basic(nn.Module):
        def forward(self, a, b, a_scale, b_scale):
            return F.scaled_mm(
                a,
                b,
                scale_a=[a_scale],
                scale_recipe_a=[F.ScalingType.BlockWise1x128],
                swizzle_a=[F.SwizzleType.NO_SWIZZLE],
                scale_b=[b_scale],
                scale_recipe_b=[F.ScalingType.BlockWise128x128],
                swizzle_b=[F.SwizzleType.NO_SWIZZLE],
                bias=None,
                output_dtype=torch.bfloat16,
            )

    a = make_fp4_tensor((128, 64))
    b = make_fp4_tensor((64, 128), stride=(1, 64))
    a_scale = torch.zeros((128, 1), dtype=torch.float32)
    b_scale = torch.zeros((4, 1), dtype=torch.float32)

    m = fx.export_and_import(
        Basic(),
        a,
        b,
        a_scale,
        b_scale,
        func_name="test_import_scaled_mm_v2_f32_blockwise_1x128_128x128_fp4",
    )
    print(m)


@run
# CHECK-LABEL: test_import_scaled_mm_v2_nv_blockwise_1x16_fp4
# CHECK: func.func @test_import_scaled_mm_v2_nv_blockwise_1x16_fp4(%arg0: !torch.vtensor<[128,64],f4E2M1FN>, %arg1: !torch.vtensor<[64,128],f4E2M1FN>, %arg2: !torch.vtensor<[1024],f8E4M3FN>, %arg3: !torch.vtensor<[1024],f8E4M3FN>) -> !torch.vtensor<[128,128],bf16>
# CHECK-NOT: torch.operator
# CHECK: %[[SCALE_A:.*]] = torch.prim.ListConstruct %arg2
# CHECK: %[[RECIPE_A_VALUE:.*]] = torch.constant.int 2
# CHECK: %[[RECIPE_A:.*]] = torch.prim.ListConstruct %[[RECIPE_A_VALUE]]
# CHECK: %[[SWIZZLE_A_VALUE:.*]] = torch.constant.int 1
# CHECK: %[[SWIZZLE_A:.*]] = torch.prim.ListConstruct %[[SWIZZLE_A_VALUE]]
# CHECK: %[[SCALE_B:.*]] = torch.prim.ListConstruct %arg3
# CHECK: %[[RECIPE_B_VALUE:.*]] = torch.constant.int 2
# CHECK: %[[RECIPE_B:.*]] = torch.prim.ListConstruct %[[RECIPE_B_VALUE]]
# CHECK: %[[SWIZZLE_B_VALUE:.*]] = torch.constant.int 1
# CHECK: %[[SWIZZLE_B:.*]] = torch.prim.ListConstruct %[[SWIZZLE_B_VALUE]]
# CHECK: %[[NONE:.*]] = torch.constant.none
# CHECK: %[[OUT_DTYPE:.*]] = torch.constant.int 15
# CHECK: %[[CONTRACTION:.*]] = torch.prim.ListConstruct
# CHECK: %[[FALSE:.*]] = torch.constant.bool false
# CHECK: %[[MM:.*]] = torch.aten._scaled_mm_v2 %arg0, %arg1, %[[SCALE_A]], %[[RECIPE_A]], %[[SWIZZLE_A]], %[[SCALE_B]], %[[RECIPE_B]], %[[SWIZZLE_B]], %[[NONE]], %[[OUT_DTYPE]], %[[CONTRACTION]], %[[FALSE]]
# CHECK: return %[[MM]]
def test_import_scaled_mm_v2_nv_blockwise_1x16_fp4():
    class Basic(nn.Module):
        def forward(self, a, b, a_scale_block, b_scale_block):
            return F.scaled_mm(
                a,
                b,
                scale_a=[a_scale_block],
                scale_recipe_a=[F.ScalingType.BlockWise1x16],
                swizzle_a=[F.SwizzleType.SWIZZLE_32_4_4],
                scale_b=[b_scale_block],
                scale_recipe_b=[F.ScalingType.BlockWise1x16],
                swizzle_b=[F.SwizzleType.SWIZZLE_32_4_4],
                bias=None,
                output_dtype=torch.bfloat16,
            )

    a = make_fp4_tensor((128, 64))
    b = make_fp4_tensor((64, 128), stride=(1, 64))
    a_scale_block = torch.zeros((1024,), dtype=torch.float8_e4m3fn)
    b_scale_block = torch.zeros((1024,), dtype=torch.float8_e4m3fn)

    m = fx.export_and_import(
        Basic(),
        a,
        b,
        a_scale_block,
        b_scale_block,
        func_name="test_import_scaled_mm_v2_nv_blockwise_1x16_fp4",
    )
    print(m)


@run
# CHECK-LABEL: test_import_scaled_mm_v2_nv_two_level_blockwise_tensorwise_fp4
# CHECK: func.func @test_import_scaled_mm_v2_nv_two_level_blockwise_tensorwise_fp4(%arg0: !torch.vtensor<[128,64],f4E2M1FN>, %arg1: !torch.vtensor<[64,128],f4E2M1FN>, %arg2: !torch.vtensor<[1024],f8E4M3FN>, %arg3: !torch.vtensor<[],f32>, %arg4: !torch.vtensor<[1024],f8E4M3FN>, %arg5: !torch.vtensor<[],f32>) -> !torch.vtensor<[128,128],bf16>
# CHECK-NOT: torch.operator
# CHECK: %[[SCALE_A:.*]] = torch.prim.ListConstruct %arg2, %arg3
# CHECK: %[[RECIPE_A_BLOCK:.*]] = torch.constant.int 2
# CHECK: %[[RECIPE_A_TENSOR:.*]] = torch.constant.int 0
# CHECK: %[[RECIPE_A:.*]] = torch.prim.ListConstruct %[[RECIPE_A_BLOCK]], %[[RECIPE_A_TENSOR]]
# CHECK: %[[SWIZZLE_A_BLOCK:.*]] = torch.constant.int 1
# CHECK: %[[SWIZZLE_A_TENSOR:.*]] = torch.constant.int 0
# CHECK: %[[SWIZZLE_A:.*]] = torch.prim.ListConstruct %[[SWIZZLE_A_BLOCK]], %[[SWIZZLE_A_TENSOR]]
# CHECK: %[[SCALE_B:.*]] = torch.prim.ListConstruct %arg4, %arg5
# CHECK: %[[RECIPE_B_BLOCK:.*]] = torch.constant.int 2
# CHECK: %[[RECIPE_B_TENSOR:.*]] = torch.constant.int 0
# CHECK: %[[RECIPE_B:.*]] = torch.prim.ListConstruct %[[RECIPE_B_BLOCK]], %[[RECIPE_B_TENSOR]]
# CHECK: %[[SWIZZLE_B_BLOCK:.*]] = torch.constant.int 1
# CHECK: %[[SWIZZLE_B_TENSOR:.*]] = torch.constant.int 0
# CHECK: %[[SWIZZLE_B:.*]] = torch.prim.ListConstruct %[[SWIZZLE_B_BLOCK]], %[[SWIZZLE_B_TENSOR]]
# CHECK: %[[NONE:.*]] = torch.constant.none
# CHECK: %[[OUT_DTYPE:.*]] = torch.constant.int 15
# CHECK: %[[CONTRACTION:.*]] = torch.prim.ListConstruct
# CHECK: %[[FALSE:.*]] = torch.constant.bool false
# CHECK: %[[MM:.*]] = torch.aten._scaled_mm_v2 %arg0, %arg1, %[[SCALE_A]], %[[RECIPE_A]], %[[SWIZZLE_A]], %[[SCALE_B]], %[[RECIPE_B]], %[[SWIZZLE_B]], %[[NONE]], %[[OUT_DTYPE]], %[[CONTRACTION]], %[[FALSE]]
# CHECK: return %[[MM]]
def test_import_scaled_mm_v2_nv_two_level_blockwise_tensorwise_fp4():
    class Basic(nn.Module):
        def forward(
            self,
            a,
            b,
            a_scale_block,
            a_scale_tensor,
            b_scale_block,
            b_scale_tensor,
        ):
            return F.scaled_mm(
                a,
                b,
                scale_a=[a_scale_block, a_scale_tensor],
                scale_recipe_a=[
                    F.ScalingType.BlockWise1x16,
                    F.ScalingType.TensorWise,
                ],
                swizzle_a=[
                    F.SwizzleType.SWIZZLE_32_4_4,
                    F.SwizzleType.NO_SWIZZLE,
                ],
                scale_b=[b_scale_block, b_scale_tensor],
                scale_recipe_b=[
                    F.ScalingType.BlockWise1x16,
                    F.ScalingType.TensorWise,
                ],
                swizzle_b=[
                    F.SwizzleType.SWIZZLE_32_4_4,
                    F.SwizzleType.NO_SWIZZLE,
                ],
                bias=None,
                output_dtype=torch.bfloat16,
            )

    a = make_fp4_tensor((128, 64))
    b = make_fp4_tensor((64, 128), stride=(1, 64))
    a_scale_block = torch.zeros((1024,), dtype=torch.float8_e4m3fn)
    b_scale_block = torch.zeros((1024,), dtype=torch.float8_e4m3fn)
    a_scale_tensor = torch.tensor(1.0, dtype=torch.float32)
    b_scale_tensor = torch.tensor(1.0, dtype=torch.float32)

    m = fx.export_and_import(
        Basic(),
        a,
        b,
        a_scale_block,
        a_scale_tensor,
        b_scale_block,
        b_scale_tensor,
        func_name="test_import_scaled_mm_v2_nv_two_level_blockwise_tensorwise_fp4",
    )
    print(m)


@run
# CHECK-LABEL: test_import_scaled_mm_v2_non_default_args_fp4
# CHECK: func.func @test_import_scaled_mm_v2_non_default_args_fp4(%arg0: !torch.vtensor<[128,64],f4E2M1FN>, %arg1: !torch.vtensor<[64,128],f4E2M1FN>, %arg2: !torch.vtensor<[512],f8E8M0FNU>, %arg3: !torch.vtensor<[512],f8E8M0FNU>, %arg4: !torch.vtensor<[128],bf16>) -> !torch.vtensor<[128,128],bf16>
# CHECK-NOT: torch.operator
# CHECK: %[[SCALE_A:.*]] = torch.prim.ListConstruct %arg2
# CHECK: %[[RECIPE_A_VALUE:.*]] = torch.constant.int 3
# CHECK: %[[RECIPE_A:.*]] = torch.prim.ListConstruct %[[RECIPE_A_VALUE]]
# CHECK: %[[SWIZZLE_A_VALUE:.*]] = torch.constant.int 1
# CHECK: %[[SWIZZLE_A:.*]] = torch.prim.ListConstruct %[[SWIZZLE_A_VALUE]]
# CHECK: %[[SCALE_B:.*]] = torch.prim.ListConstruct %arg3
# CHECK: %[[RECIPE_B_VALUE:.*]] = torch.constant.int 3
# CHECK: %[[RECIPE_B:.*]] = torch.prim.ListConstruct %[[RECIPE_B_VALUE]]
# CHECK: %[[SWIZZLE_B_VALUE:.*]] = torch.constant.int 1
# CHECK: %[[SWIZZLE_B:.*]] = torch.prim.ListConstruct %[[SWIZZLE_B_VALUE]]
# CHECK: %[[OUT_DTYPE:.*]] = torch.constant.int 15
# CHECK: %[[CONTRACTION_A_VALUE:.*]] = torch.constant.int 1
# CHECK: %[[CONTRACTION_B_VALUE:.*]] = torch.constant.int 0
# CHECK: %[[CONTRACTION:.*]] = torch.prim.ListConstruct %[[CONTRACTION_A_VALUE]], %[[CONTRACTION_B_VALUE]]
# CHECK: %[[TRUE:.*]] = torch.constant.bool true
# CHECK: %[[MM:.*]] = torch.aten._scaled_mm_v2 %arg0, %arg1, %[[SCALE_A]], %[[RECIPE_A]], %[[SWIZZLE_A]], %[[SCALE_B]], %[[RECIPE_B]], %[[SWIZZLE_B]], %arg4, %[[OUT_DTYPE]], %[[CONTRACTION]], %[[TRUE]]
# CHECK: return %[[MM]]
def test_import_scaled_mm_v2_non_default_args_fp4():
    class Basic(nn.Module):
        def forward(self, a, b, a_scale_block, b_scale_block, bias):
            return F.scaled_mm(
                a,
                b,
                scale_a=[a_scale_block],
                scale_recipe_a=[F.ScalingType.BlockWise1x32],
                swizzle_a=[F.SwizzleType.SWIZZLE_32_4_4],
                scale_b=[b_scale_block],
                scale_recipe_b=[F.ScalingType.BlockWise1x32],
                swizzle_b=[F.SwizzleType.SWIZZLE_32_4_4],
                bias=bias,
                output_dtype=torch.bfloat16,
                contraction_dim=[1, 0],
                use_fast_accum=True,
            )

    a = make_fp4_tensor((128, 64))
    b = make_fp4_tensor((64, 128), stride=(1, 64))
    a_scale_block = torch.zeros((512,), dtype=torch.float8_e8m0fnu)
    b_scale_block = torch.zeros((512,), dtype=torch.float8_e8m0fnu)
    bias = torch.zeros((128,), dtype=torch.bfloat16)

    m = fx.export_and_import(
        Basic(),
        a,
        b,
        a_scale_block,
        b_scale_block,
        bias,
        func_name="test_import_scaled_mm_v2_non_default_args_fp4",
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
