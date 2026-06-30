// RUN: torch-mlir-opt -convert-torch-to-linalg -split-input-file %s | FileCheck %s

// 2D×2D si8 symmetric (zp=0)
// CHECK-LABEL: func.func @mm_si8
// CHECK:       linalg.quantized_matmul
// CHECK-SAME:    ins({{.*}} : tensor<4x8xi8>, tensor<8x16xi8>, i32, i32)
// CHECK-SAME:    outs({{.*}} : tensor<4x16xi32>)
// CHECK-DAG:   %[[MULTIPLIER:.*]] = arith.constant 2061584302 : i64
// CHECK-DAG:   %[[HALF:.*]] = arith.constant 8589934592 : i64
// CHECK-DAG:   %[[MASK:.*]] = arith.constant 17179869183 : i64
// CHECK-DAG:   %[[SHIFT:.*]] = arith.constant 34 : i64
// CHECK-DAG:   %[[ONE:.*]] = arith.constant 1 : i64
// CHECK-DAG:   %[[ZERO64:.*]] = arith.constant 0 : i64
// CHECK-DAG:   %[[ZP:.*]] = arith.constant 0 : i32
// CHECK-DAG:   %[[QMIN:.*]] = arith.constant -128 : i32
// CHECK-DAG:   %[[QMAX:.*]] = arith.constant 127 : i32
// CHECK:       linalg.generic
// CHECK-SAME:    ins({{.*}} : tensor<4x16xi32>) outs({{.*}} : tensor<4x16xi8>)
// CHECK:         %[[ACC64:.*]] = arith.extsi {{.*}} : i32 to i64
// CHECK:         %[[PROD:.*]] = arith.muli %[[ACC64]], %[[MULTIPLIER]] : i64
// CHECK:         %[[FLOOR:.*]] = arith.shrsi %[[PROD]], %[[SHIFT]] : i64
// CHECK:         %[[REM:.*]] = arith.andi %[[PROD]], %[[MASK]] : i64
// CHECK:         %[[GT:.*]] = arith.cmpi sgt, %[[REM]], %[[HALF]] : i64
// CHECK:         %[[EQ:.*]] = arith.cmpi eq, %[[REM]], %[[HALF]] : i64
// CHECK:         %[[ODD:.*]] = arith.andi %[[FLOOR]], %[[ONE]] : i64
// CHECK:         %[[TIE:.*]] = arith.select %[[EQ]], %[[ODD]], %[[ZERO64]] : i64
// CHECK:         %[[DELTA:.*]] = arith.select %[[GT]], %[[ONE]], %[[TIE]] : i64
// CHECK:         %[[ROUNDED:.*]] = arith.addi %[[FLOOR]], %[[DELTA]] : i64
// CHECK:         arith.trunci %[[ROUNDED]] : i64 to i32
// CHECK:         arith.addi {{.*}}, %[[ZP]] : i32
// CHECK:         arith.maxsi {{.*}}, %[[QMIN]] : i32
// CHECK:         arith.minsi {{.*}}, %[[QMAX]] : i32
// CHECK:         arith.trunci {{.*}} : i32 to i8
// CHECK:       } -> tensor<4x16xi8>
// CHECK:       %[[OUT:.*]] = torch_c.from_builtin_tensor {{.*}} : tensor<4x16xi8> -> !torch.vtensor<[4,16],si8>
// CHECK:       return %[[OUT]] : !torch.vtensor<[4,16],si8>
func.func @mm_si8(
    %lhs_q: !torch.vtensor<[4,8],si8>,
    %rhs_q: !torch.vtensor<[8,16],si8>) -> !torch.vtensor<[4,16],si8> {
  %scale_lhs = torch.constant.float 3.000000e-01
  %scale_rhs = torch.constant.float 2.000000e-01
  %scale_out = torch.constant.float 5.000000e-01
  %zp   = torch.constant.int 0
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 2
  %none = torch.constant.none
  %od   = torch.derefine %none : !torch.none to !torch.optional<int>
  %lhs_f = torch.quantized_decomposed.dequantize_per_tensor
      %lhs_q, %scale_lhs, %zp, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[4,8],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
      -> !torch.vtensor<[4,8],f32>
  %rhs_f = torch.quantized_decomposed.dequantize_per_tensor
      %rhs_q, %scale_rhs, %zp, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[8,16],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
      -> !torch.vtensor<[8,16],f32>
  %result_f = torch.aten.mm %lhs_f, %rhs_f
      : !torch.vtensor<[4,8],f32>, !torch.vtensor<[8,16],f32> -> !torch.vtensor<[4,16],f32>
  %result_q = torch.quantized_decomposed.quantize_per_tensor
      %result_f, %scale_out, %zp, %qmin, %qmax, %dtype
      : !torch.vtensor<[4,16],f32>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int
      -> !torch.vtensor<[4,16],si8>
  return %result_q : !torch.vtensor<[4,16],si8>
}

// -----

// 2D×2D ui8 unsigned (zp=0), both operands get -128 sign-shift
// CHECK-LABEL: func.func @mm_ui8(
// CHECK-SAME:    %[[LHS_ARG:.*]]: !torch.vtensor<[4,8],ui8>,
// CHECK-SAME:    %[[RHS_ARG:.*]]: !torch.vtensor<[8,16],ui8>
// CHECK:       %[[LHS_RAW:.*]] = torch_c.to_builtin_tensor %[[LHS_ARG]] : !torch.vtensor<[4,8],ui8> -> tensor<4x8xi8>
// CHECK:       %[[RHS_RAW:.*]] = torch_c.to_builtin_tensor %[[RHS_ARG]] : !torch.vtensor<[8,16],ui8> -> tensor<8x16xi8>
// CHECK:       %[[LHS_SHIFTED:.*]] = linalg.generic {{.*}} ins(%[[LHS_RAW]] : tensor<4x8xi8>) outs({{.*}} : tensor<4x8xi8>)
// CHECK:         arith.addi {{.*}}, {{.*}} : i8
// CHECK:       %[[RHS_SHIFTED:.*]] = linalg.generic {{.*}} ins(%[[RHS_RAW]] : tensor<8x16xi8>) outs({{.*}} : tensor<8x16xi8>)
// CHECK:         arith.addi {{.*}}, {{.*}} : i8
// CHECK:       linalg.quantized_matmul
// CHECK-SAME:    ins(%[[LHS_SHIFTED]], %[[RHS_SHIFTED]], {{.*}} : tensor<4x8xi8>, tensor<8x16xi8>, i32, i32)
// CHECK-SAME:    outs({{.*}} : tensor<4x16xi32>)
// CHECK:       %[[QMAX_U:.*]] = arith.constant 255 : i32
// CHECK:       linalg.generic {{.*}} ins({{.*}} : tensor<4x16xi32>) outs({{.*}} : tensor<4x16xi8>)
// CHECK:         arith.extsi
// CHECK:         arith.minsi {{.*}}, %[[QMAX_U]] : i32
// CHECK:         arith.trunci {{.*}} : i32 to i8
// CHECK:       } -> tensor<4x16xi8>
// CHECK:       %[[OUT:.*]] = torch_c.from_builtin_tensor {{.*}} : tensor<4x16xi8> -> !torch.vtensor<[4,16],ui8>
// CHECK:       return %[[OUT]] : !torch.vtensor<[4,16],ui8>
func.func @mm_ui8(
    %lhs_q: !torch.vtensor<[4,8],ui8>,
    %rhs_q: !torch.vtensor<[8,16],ui8>) -> !torch.vtensor<[4,16],ui8> {
  %scale_lhs = torch.constant.float 3.921568e-03
  %scale_rhs = torch.constant.float 3.921568e-03
  %scale_out = torch.constant.float 3.921568e-03
  %zp   = torch.constant.int 0
  %qmin = torch.constant.int 0
  %qmax = torch.constant.int 255
  %dtype = torch.constant.int 0
  %none = torch.constant.none
  %od   = torch.derefine %none : !torch.none to !torch.optional<int>
  %lhs_f = torch.quantized_decomposed.dequantize_per_tensor
      %lhs_q, %scale_lhs, %zp, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[4,8],ui8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
      -> !torch.vtensor<[4,8],f32>
  %rhs_f = torch.quantized_decomposed.dequantize_per_tensor
      %rhs_q, %scale_rhs, %zp, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[8,16],ui8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
      -> !torch.vtensor<[8,16],f32>
  %result_f = torch.aten.mm %lhs_f, %rhs_f
      : !torch.vtensor<[4,8],f32>, !torch.vtensor<[8,16],f32> -> !torch.vtensor<[4,16],f32>
  %result_q = torch.quantized_decomposed.quantize_per_tensor
      %result_f, %scale_out, %zp, %qmin, %qmax, %dtype
      : !torch.vtensor<[4,16],f32>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int
      -> !torch.vtensor<[4,16],ui8>
  return %result_q : !torch.vtensor<[4,16],ui8>
}

// -----

// 2D×2D mixed: si8 lhs, ui8 rhs — lhs is raw; rhs only gets -128 sign-shift.
// CHECK-LABEL: func.func @mm_mixed_si8_ui8(
// CHECK-SAME:    %[[LHS_ARG:.*]]: !torch.vtensor<[4,8],si8>,
// CHECK-SAME:    %[[RHS_ARG:.*]]: !torch.vtensor<[8,16],ui8>
// CHECK:       %[[LHS_RAW:.*]] = torch_c.to_builtin_tensor %[[LHS_ARG]] : !torch.vtensor<[4,8],si8> -> tensor<4x8xi8>
// CHECK:       %[[RHS_RAW:.*]] = torch_c.to_builtin_tensor %[[RHS_ARG]] : !torch.vtensor<[8,16],ui8> -> tensor<8x16xi8>
// CHECK:       %[[RHS_SHIFTED:.*]] = linalg.generic {{.*}} ins(%[[RHS_RAW]] : tensor<8x16xi8>) outs({{.*}} : tensor<8x16xi8>)
// CHECK:         arith.addi {{.*}}, {{.*}} : i8
// CHECK:       linalg.quantized_matmul
// CHECK-SAME:    ins(%[[LHS_RAW]], %[[RHS_SHIFTED]], {{.*}} : tensor<4x8xi8>, tensor<8x16xi8>, i32, i32)
// CHECK-SAME:    outs({{.*}} : tensor<4x16xi32>)
// CHECK:       torch_c.from_builtin_tensor {{.*}} : tensor<4x16xi8> -> !torch.vtensor<[4,16],si8>
func.func @mm_mixed_si8_ui8(
    %lhs_q: !torch.vtensor<[4,8],si8>,
    %rhs_q: !torch.vtensor<[8,16],ui8>) -> !torch.vtensor<[4,16],si8> {
  %scale_lhs = torch.constant.float 3.000000e-01
  %scale_rhs = torch.constant.float 3.921568e-03
  %scale_out = torch.constant.float 5.000000e-01
  %zp_s  = torch.constant.int 0
  %zp_u  = torch.constant.int 0
  %qmin_s = torch.constant.int -128
  %qmax_s = torch.constant.int 127
  %qmin_u = torch.constant.int 0
  %qmax_u = torch.constant.int 255
  %dtype_s = torch.constant.int 2
  %dtype_u = torch.constant.int 0
  %none = torch.constant.none
  %od   = torch.derefine %none : !torch.none to !torch.optional<int>
  %lhs_f = torch.quantized_decomposed.dequantize_per_tensor
      %lhs_q, %scale_lhs, %zp_s, %qmin_s, %qmax_s, %dtype_s, %od
      : !torch.vtensor<[4,8],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
      -> !torch.vtensor<[4,8],f32>
  %rhs_f = torch.quantized_decomposed.dequantize_per_tensor
      %rhs_q, %scale_rhs, %zp_u, %qmin_u, %qmax_u, %dtype_u, %od
      : !torch.vtensor<[8,16],ui8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
      -> !torch.vtensor<[8,16],f32>
  %result_f = torch.aten.mm %lhs_f, %rhs_f
      : !torch.vtensor<[4,8],f32>, !torch.vtensor<[8,16],f32> -> !torch.vtensor<[4,16],f32>
  %result_q = torch.quantized_decomposed.quantize_per_tensor
      %result_f, %scale_out, %zp_s, %qmin_s, %qmax_s, %dtype_s
      : !torch.vtensor<[4,16],f32>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int
      -> !torch.vtensor<[4,16],si8>
  return %result_q : !torch.vtensor<[4,16],si8>
}

// -----

// 2D×2D si8 via aten.matmul (exercises ConvertAtenMatmulQDQFusionOp, not ConvertAtenMmQDQFusionOp)
// CHECK-LABEL: func.func @matmul_si8_2d
// CHECK:       linalg.quantized_matmul
// CHECK-SAME:    ins({{.*}} : tensor<4x8xi8>, tensor<8x16xi8>, i32, i32)
// CHECK-SAME:    outs({{.*}} : tensor<4x16xi32>)
// CHECK:       torch_c.from_builtin_tensor {{.*}} : tensor<4x16xi8> -> !torch.vtensor<[4,16],si8>
func.func @matmul_si8_2d(
    %lhs_q: !torch.vtensor<[4,8],si8>,
    %rhs_q: !torch.vtensor<[8,16],si8>) -> !torch.vtensor<[4,16],si8> {
  %scale_lhs = torch.constant.float 3.000000e-01
  %scale_rhs = torch.constant.float 2.000000e-01
  %scale_out = torch.constant.float 5.000000e-01
  %zp   = torch.constant.int 0
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 2
  %none = torch.constant.none
  %od   = torch.derefine %none : !torch.none to !torch.optional<int>
  %lhs_f = torch.quantized_decomposed.dequantize_per_tensor
      %lhs_q, %scale_lhs, %zp, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[4,8],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
      -> !torch.vtensor<[4,8],f32>
  %rhs_f = torch.quantized_decomposed.dequantize_per_tensor
      %rhs_q, %scale_rhs, %zp, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[8,16],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
      -> !torch.vtensor<[8,16],f32>
  %result_f = torch.aten.matmul %lhs_f, %rhs_f
      : !torch.vtensor<[4,8],f32>, !torch.vtensor<[8,16],f32> -> !torch.vtensor<[4,16],f32>
  %result_q = torch.quantized_decomposed.quantize_per_tensor
      %result_f, %scale_out, %zp, %qmin, %qmax, %dtype
      : !torch.vtensor<[4,16],f32>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int
      -> !torch.vtensor<[4,16],si8>
  return %result_q : !torch.vtensor<[4,16],si8>
}

// -----

// 3D×3D batched si8 via aten.matmul (exercises ConvertAtenMatmulQDQFusionOp 3D path)
// CHECK-LABEL: func.func @bmm_si8
// CHECK:       linalg.quantized_batch_matmul
// CHECK-SAME:    ins({{.*}} : tensor<2x4x8xi8>, tensor<2x8x16xi8>, i32, i32)
// CHECK-SAME:    outs({{.*}} : tensor<2x4x16xi32>)
// CHECK:       torch_c.from_builtin_tensor {{.*}} : tensor<2x4x16xi8> -> !torch.vtensor<[2,4,16],si8>
func.func @bmm_si8(
    %lhs_q: !torch.vtensor<[2,4,8],si8>,
    %rhs_q: !torch.vtensor<[2,8,16],si8>) -> !torch.vtensor<[2,4,16],si8> {
  %scale_lhs = torch.constant.float 3.000000e-01
  %scale_rhs = torch.constant.float 2.000000e-01
  %scale_out = torch.constant.float 5.000000e-01
  %zp   = torch.constant.int 0
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 2
  %none = torch.constant.none
  %od   = torch.derefine %none : !torch.none to !torch.optional<int>
  %lhs_f = torch.quantized_decomposed.dequantize_per_tensor
      %lhs_q, %scale_lhs, %zp, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[2,4,8],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
      -> !torch.vtensor<[2,4,8],f32>
  %rhs_f = torch.quantized_decomposed.dequantize_per_tensor
      %rhs_q, %scale_rhs, %zp, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[2,8,16],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
      -> !torch.vtensor<[2,8,16],f32>
  %result_f = torch.aten.matmul %lhs_f, %rhs_f
      : !torch.vtensor<[2,4,8],f32>, !torch.vtensor<[2,8,16],f32> -> !torch.vtensor<[2,4,16],f32>
  %result_q = torch.quantized_decomposed.quantize_per_tensor
      %result_f, %scale_out, %zp, %qmin, %qmax, %dtype
      : !torch.vtensor<[2,4,16],f32>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int
      -> !torch.vtensor<[2,4,16],si8>
  return %result_q : !torch.vtensor<[2,4,16],si8>
}

// -----

// 3D×3D batched si8 via aten.bmm (exercises ConvertAtenBmmQDQFusionOp)
// CHECK-LABEL: func.func @bmm_via_aten_bmm_si8
// CHECK:       linalg.quantized_batch_matmul
// CHECK-SAME:    ins({{.*}} : tensor<2x4x8xi8>, tensor<2x8x16xi8>, i32, i32)
// CHECK-SAME:    outs({{.*}} : tensor<2x4x16xi32>)
// CHECK:       torch_c.from_builtin_tensor {{.*}} : tensor<2x4x16xi8> -> !torch.vtensor<[2,4,16],si8>
func.func @bmm_via_aten_bmm_si8(
    %lhs_q: !torch.vtensor<[2,4,8],si8>,
    %rhs_q: !torch.vtensor<[2,8,16],si8>) -> !torch.vtensor<[2,4,16],si8> {
  %scale_lhs = torch.constant.float 3.000000e-01
  %scale_rhs = torch.constant.float 2.000000e-01
  %scale_out = torch.constant.float 5.000000e-01
  %zp   = torch.constant.int 0
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 2
  %none = torch.constant.none
  %od   = torch.derefine %none : !torch.none to !torch.optional<int>
  %lhs_f = torch.quantized_decomposed.dequantize_per_tensor
      %lhs_q, %scale_lhs, %zp, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[2,4,8],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
      -> !torch.vtensor<[2,4,8],f32>
  %rhs_f = torch.quantized_decomposed.dequantize_per_tensor
      %rhs_q, %scale_rhs, %zp, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[2,8,16],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
      -> !torch.vtensor<[2,8,16],f32>
  %result_f = torch.aten.bmm %lhs_f, %rhs_f
      : !torch.vtensor<[2,4,8],f32>, !torch.vtensor<[2,8,16],f32> -> !torch.vtensor<[2,4,16],f32>
  %result_q = torch.quantized_decomposed.quantize_per_tensor
      %result_f, %scale_out, %zp, %qmin, %qmax, %dtype
      : !torch.vtensor<[2,4,16],f32>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int
      -> !torch.vtensor<[2,4,16],si8>
  return %result_q : !torch.vtensor<[2,4,16],si8>
}

// -----

// 1D×2D vec-mat si8 — lhs unsqueezed to [1,8], result collapsed back to [16].
// CHECK-LABEL: func.func @vecmat_si8
// CHECK:       tensor.expand_shape {{.*}} : tensor<8xi8> into tensor<1x8xi8>
// CHECK:       linalg.quantized_matmul
// CHECK-SAME:    ins({{.*}} : tensor<1x8xi8>, tensor<8x16xi8>, i32, i32)
// CHECK-SAME:    outs({{.*}} : tensor<1x16xi32>)
// CHECK:       tensor.collapse_shape {{.*}} : tensor<1x16xi32> into tensor<16xi32>
// CHECK:       torch_c.from_builtin_tensor {{.*}} : tensor<16xi8> -> !torch.vtensor<[16],si8>
func.func @vecmat_si8(
    %lhs_q: !torch.vtensor<[8],si8>,
    %rhs_q: !torch.vtensor<[8,16],si8>) -> !torch.vtensor<[16],si8> {
  %scale_lhs = torch.constant.float 3.000000e-01
  %scale_rhs = torch.constant.float 2.000000e-01
  %scale_out = torch.constant.float 5.000000e-01
  %zp   = torch.constant.int 0
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 2
  %none = torch.constant.none
  %od   = torch.derefine %none : !torch.none to !torch.optional<int>
  %lhs_f = torch.quantized_decomposed.dequantize_per_tensor
      %lhs_q, %scale_lhs, %zp, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[8],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
      -> !torch.vtensor<[8],f32>
  %rhs_f = torch.quantized_decomposed.dequantize_per_tensor
      %rhs_q, %scale_rhs, %zp, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[8,16],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
      -> !torch.vtensor<[8,16],f32>
  %result_f = torch.aten.matmul %lhs_f, %rhs_f
      : !torch.vtensor<[8],f32>, !torch.vtensor<[8,16],f32> -> !torch.vtensor<[16],f32>
  %result_q = torch.quantized_decomposed.quantize_per_tensor
      %result_f, %scale_out, %zp, %qmin, %qmax, %dtype
      : !torch.vtensor<[16],f32>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int
      -> !torch.vtensor<[16],si8>
  return %result_q : !torch.vtensor<[16],si8>
}

// -----

// 2D×1D mat-vec si8 — rhs unsqueezed to [8,1], result collapsed to [4].
// CHECK-LABEL: func.func @matvec_si8
// CHECK:       tensor.expand_shape {{.*}} : tensor<8xi8> into tensor<8x1xi8>
// CHECK:       linalg.quantized_matmul
// CHECK-SAME:    ins({{.*}} : tensor<4x8xi8>, tensor<8x1xi8>, i32, i32)
// CHECK-SAME:    outs({{.*}} : tensor<4x1xi32>)
// CHECK:       tensor.collapse_shape {{.*}} : tensor<4x1xi32> into tensor<4xi32>
// CHECK:       torch_c.from_builtin_tensor {{.*}} : tensor<4xi8> -> !torch.vtensor<[4],si8>
func.func @matvec_si8(
    %lhs_q: !torch.vtensor<[4,8],si8>,
    %rhs_q: !torch.vtensor<[8],si8>) -> !torch.vtensor<[4],si8> {
  %scale_lhs = torch.constant.float 3.000000e-01
  %scale_rhs = torch.constant.float 2.000000e-01
  %scale_out = torch.constant.float 5.000000e-01
  %zp   = torch.constant.int 0
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 2
  %none = torch.constant.none
  %od   = torch.derefine %none : !torch.none to !torch.optional<int>
  %lhs_f = torch.quantized_decomposed.dequantize_per_tensor
      %lhs_q, %scale_lhs, %zp, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[4,8],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
      -> !torch.vtensor<[4,8],f32>
  %rhs_f = torch.quantized_decomposed.dequantize_per_tensor
      %rhs_q, %scale_rhs, %zp, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[8],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
      -> !torch.vtensor<[8],f32>
  %result_f = torch.aten.matmul %lhs_f, %rhs_f
      : !torch.vtensor<[4,8],f32>, !torch.vtensor<[8],f32> -> !torch.vtensor<[4],f32>
  %result_q = torch.quantized_decomposed.quantize_per_tensor
      %result_f, %scale_out, %zp, %qmin, %qmax, %dtype
      : !torch.vtensor<[4],f32>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int
      -> !torch.vtensor<[4],si8>
  return %result_q : !torch.vtensor<[4],si8>
}

// -----

// 1D×1D vec-vec si8 dot product — both inputs unsqueezed, result collapsed to scalar.
// CHECK-LABEL: func.func @vecvec_si8
// CHECK:       tensor.expand_shape {{.*}} : tensor<8xi8> into tensor<1x8xi8>
// CHECK:       tensor.expand_shape {{.*}} : tensor<8xi8> into tensor<8x1xi8>
// CHECK:       linalg.quantized_matmul
// CHECK-SAME:    ins({{.*}} : tensor<1x8xi8>, tensor<8x1xi8>, i32, i32)
// CHECK-SAME:    outs({{.*}} : tensor<1x1xi32>)
// CHECK:       tensor.collapse_shape {{.*}} : tensor<1x1xi32> into tensor<i32>
// CHECK:       arith.shrsi
// CHECK:       arith.trunci {{.*}} : i32 to i8
// CHECK:       torch_c.from_builtin_tensor {{.*}} : tensor<i8> -> !torch.vtensor<[],si8>
func.func @vecvec_si8(
    %lhs_q: !torch.vtensor<[8],si8>,
    %rhs_q: !torch.vtensor<[8],si8>) -> !torch.vtensor<[],si8> {
  %scale_lhs = torch.constant.float 3.000000e-01
  %scale_rhs = torch.constant.float 2.000000e-01
  %scale_out = torch.constant.float 5.000000e-01
  %zp   = torch.constant.int 0
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 2
  %none = torch.constant.none
  %od   = torch.derefine %none : !torch.none to !torch.optional<int>
  %lhs_f = torch.quantized_decomposed.dequantize_per_tensor
      %lhs_q, %scale_lhs, %zp, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[8],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
      -> !torch.vtensor<[8],f32>
  %rhs_f = torch.quantized_decomposed.dequantize_per_tensor
      %rhs_q, %scale_rhs, %zp, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[8],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
      -> !torch.vtensor<[8],f32>
  %result_f = torch.aten.matmul %lhs_f, %rhs_f
      : !torch.vtensor<[8],f32>, !torch.vtensor<[8],f32> -> !torch.vtensor<[],f32>
  %result_q = torch.quantized_decomposed.quantize_per_tensor
      %result_f, %scale_out, %zp, %qmin, %qmax, %dtype
      : !torch.vtensor<[],f32>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int
      -> !torch.vtensor<[],si8>
  return %result_q : !torch.vtensor<[],si8>
}

// -----

// mm result has two uses — fusion bails, falls back to float linalg.matmul.
// dq/q are lowered to linalg.generic by ConvertElementwiseOp.
// CHECK-LABEL: func.func @mm_si8_no_fuse_multi_use_mm
// CHECK-NOT: linalg.quantized_matmul
// CHECK-NOT: torch.quantized_decomposed
// CHECK: linalg.generic
// CHECK: linalg.matmul
func.func @mm_si8_no_fuse_multi_use_mm(
    %lhs_q: !torch.vtensor<[4,8],si8>,
    %rhs_q: !torch.vtensor<[8,16],si8>) -> (!torch.vtensor<[4,16],si8>, !torch.vtensor<[4,16],f32>) {
  %scale_lhs = torch.constant.float 3.000000e-01
  %scale_rhs = torch.constant.float 2.000000e-01
  %scale_out = torch.constant.float 5.000000e-01
  %zp   = torch.constant.int 0
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 2
  %none = torch.constant.none
  %od   = torch.derefine %none : !torch.none to !torch.optional<int>
  %lhs_f = torch.quantized_decomposed.dequantize_per_tensor
      %lhs_q, %scale_lhs, %zp, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[4,8],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
      -> !torch.vtensor<[4,8],f32>
  %rhs_f = torch.quantized_decomposed.dequantize_per_tensor
      %rhs_q, %scale_rhs, %zp, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[8,16],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
      -> !torch.vtensor<[8,16],f32>
  %result_f = torch.aten.mm %lhs_f, %rhs_f
      : !torch.vtensor<[4,8],f32>, !torch.vtensor<[8,16],f32> -> !torch.vtensor<[4,16],f32>
  %result_q = torch.quantized_decomposed.quantize_per_tensor
      %result_f, %scale_out, %zp, %qmin, %qmax, %dtype
      : !torch.vtensor<[4,16],f32>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int
      -> !torch.vtensor<[4,16],si8>
  // result_f has two uses: quantize_per_tensor and a direct return.
  return %result_q, %result_f : !torch.vtensor<[4,16],si8>, !torch.vtensor<[4,16],f32>
}

// -----

// Inputs not from dequantize_per_tensor — fusion bails, falls back to float linalg.matmul.
// No dq ops present; q op is lowered to linalg.generic by ConvertElementwiseOp.
// CHECK-LABEL: func.func @mm_no_fuse_plain_f32_inputs
// CHECK-NOT: linalg.quantized_matmul
// CHECK-NOT: torch.quantized_decomposed
// CHECK: linalg.matmul
// CHECK: linalg.generic
func.func @mm_no_fuse_plain_f32_inputs(
    %lhs_f: !torch.vtensor<[4,8],f32>,
    %rhs_f: !torch.vtensor<[8,16],f32>) -> !torch.vtensor<[4,16],si8> {
  %scale_out = torch.constant.float 5.000000e-01
  %zp   = torch.constant.int 0
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 2
  %result_f = torch.aten.mm %lhs_f, %rhs_f
      : !torch.vtensor<[4,8],f32>, !torch.vtensor<[8,16],f32> -> !torch.vtensor<[4,16],f32>
  %result_q = torch.quantized_decomposed.quantize_per_tensor
      %result_f, %scale_out, %zp, %qmin, %qmax, %dtype
      : !torch.vtensor<[4,16],f32>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int
      -> !torch.vtensor<[4,16],si8>
  return %result_q : !torch.vtensor<[4,16],si8>
}

// -----

// 4D×4D quantized bmm with two dynamic batch dims — fusion bails (no named op),
// falls back to float linalg.generic via ConvertAtenMatmulOp.
// CHECK-LABEL: func.func @bmm_si8_two_dynamic_batch_dims
// CHECK-NOT: linalg.quantized_batch_matmul
// CHECK-NOT: torch.quantized_decomposed
// CHECK: linalg.generic
func.func @bmm_si8_two_dynamic_batch_dims(
    %lhs_q: !torch.vtensor<[?,?,4,8],si8>,
    %rhs_q: !torch.vtensor<[?,?,8,16],si8>) -> !torch.vtensor<[?,?,4,16],si8> {
  %scale_lhs = torch.constant.float 3.000000e-01
  %scale_rhs = torch.constant.float 2.000000e-01
  %scale_out = torch.constant.float 5.000000e-01
  %zp   = torch.constant.int 0
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 2
  %none = torch.constant.none
  %od   = torch.derefine %none : !torch.none to !torch.optional<int>
  %lhs_f = torch.quantized_decomposed.dequantize_per_tensor
      %lhs_q, %scale_lhs, %zp, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[?,?,4,8],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
      -> !torch.vtensor<[?,?,4,8],f32>
  %rhs_f = torch.quantized_decomposed.dequantize_per_tensor
      %rhs_q, %scale_rhs, %zp, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[?,?,8,16],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
      -> !torch.vtensor<[?,?,8,16],f32>
  %result_f = torch.aten.matmul %lhs_f, %rhs_f
      : !torch.vtensor<[?,?,4,8],f32>, !torch.vtensor<[?,?,8,16],f32> -> !torch.vtensor<[?,?,4,16],f32>
  %result_q = torch.quantized_decomposed.quantize_per_tensor
      %result_f, %scale_out, %zp, %qmin, %qmax, %dtype
      : !torch.vtensor<[?,?,4,16],f32>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int
      -> !torch.vtensor<[?,?,4,16],si8>
  return %result_q : !torch.vtensor<[?,?,4,16],si8>
}

// -----

// Effective scale (lhsScale * rhsScale / outScale = 1e-20) is out of the Q31
// representable range, so the fusion's quantizeMultiplier would fail. The
// legality predicate must reject the chain up front, so the chain falls back cleanly
// to the float decomposed route with no stranded torch.quantized_decomposed.
// CHECK-LABEL: func.func @mm_si8_no_fuse_scale_out_of_q31_range
// CHECK-NOT:   linalg.quantized_matmul
// CHECK-NOT:   torch.quantized_decomposed
// CHECK:       linalg.generic
func.func @mm_si8_no_fuse_scale_out_of_q31_range(
    %lhs_q: !torch.vtensor<[4,8],si8>,
    %rhs_q: !torch.vtensor<[8,16],si8>) -> !torch.vtensor<[4,16],si8> {
  %scale_lhs = torch.constant.float 1.000000e-10
  %scale_rhs = torch.constant.float 1.000000e-10
  %scale_out = torch.constant.float 1.000000e+00
  %zp   = torch.constant.int 0
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 2
  %none = torch.constant.none
  %od   = torch.derefine %none : !torch.none to !torch.optional<int>
  %lhs_f = torch.quantized_decomposed.dequantize_per_tensor
      %lhs_q, %scale_lhs, %zp, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[4,8],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
      -> !torch.vtensor<[4,8],f32>
  %rhs_f = torch.quantized_decomposed.dequantize_per_tensor
      %rhs_q, %scale_rhs, %zp, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[8,16],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
      -> !torch.vtensor<[8,16],f32>
  %result_f = torch.aten.mm %lhs_f, %rhs_f
      : !torch.vtensor<[4,8],f32>, !torch.vtensor<[8,16],f32> -> !torch.vtensor<[4,16],f32>
  %result_q = torch.quantized_decomposed.quantize_per_tensor
      %result_f, %scale_out, %zp, %qmin, %qmax, %dtype
      : !torch.vtensor<[4,16],f32>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int
      -> !torch.vtensor<[4,16],si8>
  return %result_q : !torch.vtensor<[4,16],si8>
}

// -----

// Non-constant lhs scale — fusion bails, falls back to float linalg.matmul.
// dq/q are lowered to linalg.generic by ConvertElementwiseOp.
// CHECK-LABEL: func.func @mm_si8_no_fuse_dynamic_scale
// CHECK-NOT: linalg.quantized_matmul
// CHECK-NOT: torch.quantized_decomposed
// CHECK: linalg.generic
// CHECK: linalg.matmul
func.func @mm_si8_no_fuse_dynamic_scale(
    %lhs_q: !torch.vtensor<[4,8],si8>,
    %rhs_q: !torch.vtensor<[8,16],si8>,
    %scale_lhs: !torch.float) -> !torch.vtensor<[4,16],si8> {
  %scale_rhs = torch.constant.float 2.000000e-01
  %scale_out = torch.constant.float 5.000000e-01
  %zp   = torch.constant.int 0
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 2
  %none = torch.constant.none
  %od   = torch.derefine %none : !torch.none to !torch.optional<int>
  %lhs_f = torch.quantized_decomposed.dequantize_per_tensor
      %lhs_q, %scale_lhs, %zp, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[4,8],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
      -> !torch.vtensor<[4,8],f32>
  %rhs_f = torch.quantized_decomposed.dequantize_per_tensor
      %rhs_q, %scale_rhs, %zp, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[8,16],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
      -> !torch.vtensor<[8,16],f32>
  %result_f = torch.aten.mm %lhs_f, %rhs_f
      : !torch.vtensor<[4,8],f32>, !torch.vtensor<[8,16],f32> -> !torch.vtensor<[4,16],f32>
  %result_q = torch.quantized_decomposed.quantize_per_tensor
      %result_f, %scale_out, %zp, %qmin, %qmax, %dtype
      : !torch.vtensor<[4,16],f32>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int
      -> !torch.vtensor<[4,16],si8>
  return %result_q : !torch.vtensor<[4,16],si8>
}

// -----

// lhs dq has a non-mm user (direct return) — fusion still fires; lhs dq is
// also lowered to linalg.generic by ConvertElementwiseOp for the extra use.
// CHECK-LABEL: func.func @mm_si8_lhs_dq_shared
// CHECK:       linalg.quantized_matmul
// CHECK-SAME:    ins({{.*}} : tensor<4x8xi8>, tensor<8x16xi8>, i32, i32)
// CHECK-SAME:    outs({{.*}} : tensor<4x16xi32>)
// CHECK:       linalg.generic
// CHECK:       torch_c.from_builtin_tensor {{.*}} : tensor<4x16xi8> -> !torch.vtensor<[4,16],si8>
func.func @mm_si8_lhs_dq_shared(
    %lhs_q: !torch.vtensor<[4,8],si8>,
    %rhs_q: !torch.vtensor<[8,16],si8>) -> (!torch.vtensor<[4,16],si8>, !torch.vtensor<[4,8],f32>) {
  %scale_lhs = torch.constant.float 3.000000e-01
  %scale_rhs = torch.constant.float 2.000000e-01
  %scale_out = torch.constant.float 5.000000e-01
  %zp   = torch.constant.int 0
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 2
  %none = torch.constant.none
  %od   = torch.derefine %none : !torch.none to !torch.optional<int>
  %lhs_f = torch.quantized_decomposed.dequantize_per_tensor
      %lhs_q, %scale_lhs, %zp, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[4,8],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
      -> !torch.vtensor<[4,8],f32>
  %rhs_f = torch.quantized_decomposed.dequantize_per_tensor
      %rhs_q, %scale_rhs, %zp, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[8,16],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
      -> !torch.vtensor<[8,16],f32>
  %result_f = torch.aten.mm %lhs_f, %rhs_f
      : !torch.vtensor<[4,8],f32>, !torch.vtensor<[8,16],f32> -> !torch.vtensor<[4,16],f32>
  %result_q = torch.quantized_decomposed.quantize_per_tensor
      %result_f, %scale_out, %zp, %qmin, %qmax, %dtype
      : !torch.vtensor<[4,16],f32>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int
      -> !torch.vtensor<[4,16],si8>
  // lhs_f has two users: mm (fused) and this direct return (lowered by ConvertElementwiseOp).
  return %result_q, %lhs_f : !torch.vtensor<[4,16],si8>, !torch.vtensor<[4,8],f32>
}

// -----

// Same dq feeds both operands of mm (lhsDq == rhsDq). Fusion must fire and
// erase the dq exactly once.
// CHECK-LABEL: func.func @mm_si8_self_dq
// CHECK-NOT:   torch.quantized_decomposed
// CHECK-NOT:   linalg.matmul
// CHECK:       linalg.quantized_matmul
// CHECK-SAME:    ins({{.*}} : tensor<4x4xi8>, tensor<4x4xi8>, i32, i32)
// CHECK-SAME:    outs({{.*}} : tensor<4x4xi32>)
// CHECK:       torch_c.from_builtin_tensor {{.*}} : tensor<4x4xi8> -> !torch.vtensor<[4,4],si8>
func.func @mm_si8_self_dq(
    %x_q: !torch.vtensor<[4,4],si8>) -> !torch.vtensor<[4,4],si8> {
  %scale = torch.constant.float 3.000000e-01
  %scale_out = torch.constant.float 5.000000e-01
  %zp   = torch.constant.int 0
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 2
  %none = torch.constant.none
  %od   = torch.derefine %none : !torch.none to !torch.optional<int>
  %x_f = torch.quantized_decomposed.dequantize_per_tensor
      %x_q, %scale, %zp, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[4,4],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
      -> !torch.vtensor<[4,4],f32>
  %result_f = torch.aten.mm %x_f, %x_f
      : !torch.vtensor<[4,4],f32>, !torch.vtensor<[4,4],f32> -> !torch.vtensor<[4,4],f32>
  %result_q = torch.quantized_decomposed.quantize_per_tensor
      %result_f, %scale_out, %zp, %qmin, %qmax, %dtype
      : !torch.vtensor<[4,4],f32>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int
      -> !torch.vtensor<[4,4],si8>
  return %result_q : !torch.vtensor<[4,4],si8>
}

// -----

// Non-constant output scale — fusion bails, falls back to float linalg.matmul.
// dq/q are lowered to linalg.generic by ConvertElementwiseOp.
// CHECK-LABEL: func.func @mm_si8_no_fuse_dynamic_out_scale
// CHECK-NOT: linalg.quantized_matmul
// CHECK-NOT: torch.quantized_decomposed
// CHECK: linalg.generic
// CHECK: linalg.matmul
func.func @mm_si8_no_fuse_dynamic_out_scale(
    %lhs_q: !torch.vtensor<[4,8],si8>,
    %rhs_q: !torch.vtensor<[8,16],si8>,
    %scale_out: !torch.float) -> !torch.vtensor<[4,16],si8> {
  %scale_lhs = torch.constant.float 3.000000e-01
  %scale_rhs = torch.constant.float 2.000000e-01
  %zp   = torch.constant.int 0
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 2
  %none = torch.constant.none
  %od   = torch.derefine %none : !torch.none to !torch.optional<int>
  %lhs_f = torch.quantized_decomposed.dequantize_per_tensor
      %lhs_q, %scale_lhs, %zp, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[4,8],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
      -> !torch.vtensor<[4,8],f32>
  %rhs_f = torch.quantized_decomposed.dequantize_per_tensor
      %rhs_q, %scale_rhs, %zp, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[8,16],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
      -> !torch.vtensor<[8,16],f32>
  %result_f = torch.aten.mm %lhs_f, %rhs_f
      : !torch.vtensor<[4,8],f32>, !torch.vtensor<[8,16],f32> -> !torch.vtensor<[4,16],f32>
  %result_q = torch.quantized_decomposed.quantize_per_tensor
      %result_f, %scale_out, %zp, %qmin, %qmax, %dtype
      : !torch.vtensor<[4,16],f32>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int
      -> !torch.vtensor<[4,16],si8>
  return %result_q : !torch.vtensor<[4,16],si8>
}

// -----

// 1D×1D dynamic dot product
// CHECK-LABEL: func.func @vecvec_si8_dynamic
// CHECK:       tensor.expand_shape {{.*}} : tensor<?xi8> into tensor<1x?xi8>
// CHECK:       tensor.expand_shape {{.*}} : tensor<?xi8> into tensor<?x1xi8>
// CHECK:       linalg.quantized_matmul
// CHECK-SAME:    ins({{.*}} : tensor<1x?xi8>, tensor<?x1xi8>, i32, i32)
// CHECK-SAME:    outs({{.*}} : tensor<1x1xi32>)
// CHECK:       tensor.collapse_shape {{.*}} : tensor<1x1xi32> into tensor<i32>
// CHECK:       torch_c.from_builtin_tensor {{.*}} : tensor<i8> -> !torch.vtensor<[],si8>
func.func @vecvec_si8_dynamic(
    %lhs_q: !torch.vtensor<[?],si8>,
    %rhs_q: !torch.vtensor<[?],si8>) -> !torch.vtensor<[],si8> {
  %scale_lhs = torch.constant.float 3.000000e-01
  %scale_rhs = torch.constant.float 2.000000e-01
  %scale_out = torch.constant.float 5.000000e-01
  %zp   = torch.constant.int 0
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 2
  %none = torch.constant.none
  %od   = torch.derefine %none : !torch.none to !torch.optional<int>
  %lhs_f = torch.quantized_decomposed.dequantize_per_tensor
      %lhs_q, %scale_lhs, %zp, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[?],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
      -> !torch.vtensor<[?],f32>
  %rhs_f = torch.quantized_decomposed.dequantize_per_tensor
      %rhs_q, %scale_rhs, %zp, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[?],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
      -> !torch.vtensor<[?],f32>
  %result_f = torch.aten.matmul %lhs_f, %rhs_f
      : !torch.vtensor<[?],f32>, !torch.vtensor<[?],f32> -> !torch.vtensor<[],f32>
  %result_q = torch.quantized_decomposed.quantize_per_tensor
      %result_f, %scale_out, %zp, %qmin, %qmax, %dtype
      : !torch.vtensor<[],f32>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int
      -> !torch.vtensor<[],si8>
  return %result_q : !torch.vtensor<[],si8>
}

// -----

// 3D×2D matmul: the fusion body only handles (1,1),(2,2),(1,2),(2,1),(>2,>2),
// so matchQDQMatmulChain must reject this rank pair. Must fall back cleanly to the
// float decomposed route (dq/q lowered to linalg.generic, no quantized op).
// CHECK-LABEL: func.func @matmul_si8_3d_2d
// CHECK-NOT:   linalg.quantized_matmul
// CHECK-NOT:   torch.quantized_decomposed
// CHECK:       linalg.generic
func.func @matmul_si8_3d_2d(
    %lhs_q: !torch.vtensor<[2,4,8],si8>,
    %rhs_q: !torch.vtensor<[8,16],si8>) -> !torch.vtensor<[2,4,16],si8> {
  %scale_lhs = torch.constant.float 3.000000e-01
  %scale_rhs = torch.constant.float 2.000000e-01
  %scale_out = torch.constant.float 5.000000e-01
  %zp   = torch.constant.int 0
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 2
  %none = torch.constant.none
  %od   = torch.derefine %none : !torch.none to !torch.optional<int>
  %lhs_f = torch.quantized_decomposed.dequantize_per_tensor
      %lhs_q, %scale_lhs, %zp, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[2,4,8],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
      -> !torch.vtensor<[2,4,8],f32>
  %rhs_f = torch.quantized_decomposed.dequantize_per_tensor
      %rhs_q, %scale_rhs, %zp, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[8,16],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
      -> !torch.vtensor<[8,16],f32>
  %result_f = torch.aten.matmul %lhs_f, %rhs_f
      : !torch.vtensor<[2,4,8],f32>, !torch.vtensor<[8,16],f32> -> !torch.vtensor<[2,4,16],f32>
  %result_q = torch.quantized_decomposed.quantize_per_tensor
      %result_f, %scale_out, %zp, %qmin, %qmax, %dtype
      : !torch.vtensor<[2,4,16],f32>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int
      -> !torch.vtensor<[2,4,16],si8>
  return %result_q : !torch.vtensor<[2,4,16],si8>
}

// -----

// 2D×3D matmul: same unsupported-rank fallback as the 3D×2D case above.
// CHECK-LABEL: func.func @matmul_si8_2d_3d
// CHECK-NOT:   linalg.quantized_matmul
// CHECK-NOT:   torch.quantized_decomposed
// CHECK:       linalg.generic
func.func @matmul_si8_2d_3d(
    %lhs_q: !torch.vtensor<[4,8],si8>,
    %rhs_q: !torch.vtensor<[2,8,16],si8>) -> !torch.vtensor<[2,4,16],si8> {
  %scale_lhs = torch.constant.float 3.000000e-01
  %scale_rhs = torch.constant.float 2.000000e-01
  %scale_out = torch.constant.float 5.000000e-01
  %zp   = torch.constant.int 0
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 2
  %none = torch.constant.none
  %od   = torch.derefine %none : !torch.none to !torch.optional<int>
  %lhs_f = torch.quantized_decomposed.dequantize_per_tensor
      %lhs_q, %scale_lhs, %zp, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[4,8],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
      -> !torch.vtensor<[4,8],f32>
  %rhs_f = torch.quantized_decomposed.dequantize_per_tensor
      %rhs_q, %scale_rhs, %zp, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[2,8,16],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
      -> !torch.vtensor<[2,8,16],f32>
  %result_f = torch.aten.matmul %lhs_f, %rhs_f
      : !torch.vtensor<[4,8],f32>, !torch.vtensor<[2,8,16],f32> -> !torch.vtensor<[2,4,16],f32>
  %result_q = torch.quantized_decomposed.quantize_per_tensor
      %result_f, %scale_out, %zp, %qmin, %qmax, %dtype
      : !torch.vtensor<[2,4,16],f32>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int
      -> !torch.vtensor<[2,4,16],si8>
  return %result_q : !torch.vtensor<[2,4,16],si8>
}

// -----

// 4D×2D matmul: higher-rank lhs against a 2D rhs is also unsupported by the
// fusion body and must fall back to the float decomposed route.
// CHECK-LABEL: func.func @matmul_si8_4d_2d
// CHECK-NOT:   linalg.quantized_matmul
// CHECK-NOT:   torch.quantized_decomposed
// CHECK:       linalg.generic
func.func @matmul_si8_4d_2d(
    %lhs_q: !torch.vtensor<[3,2,4,8],si8>,
    %rhs_q: !torch.vtensor<[8,16],si8>) -> !torch.vtensor<[3,2,4,16],si8> {
  %scale_lhs = torch.constant.float 3.000000e-01
  %scale_rhs = torch.constant.float 2.000000e-01
  %scale_out = torch.constant.float 5.000000e-01
  %zp   = torch.constant.int 0
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 2
  %none = torch.constant.none
  %od   = torch.derefine %none : !torch.none to !torch.optional<int>
  %lhs_f = torch.quantized_decomposed.dequantize_per_tensor
      %lhs_q, %scale_lhs, %zp, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[3,2,4,8],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
      -> !torch.vtensor<[3,2,4,8],f32>
  %rhs_f = torch.quantized_decomposed.dequantize_per_tensor
      %rhs_q, %scale_rhs, %zp, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[8,16],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
      -> !torch.vtensor<[8,16],f32>
  %result_f = torch.aten.matmul %lhs_f, %rhs_f
      : !torch.vtensor<[3,2,4,8],f32>, !torch.vtensor<[8,16],f32> -> !torch.vtensor<[3,2,4,16],f32>
  %result_q = torch.quantized_decomposed.quantize_per_tensor
      %result_f, %scale_out, %zp, %qmin, %qmax, %dtype
      : !torch.vtensor<[3,2,4,16],f32>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int
      -> !torch.vtensor<[3,2,4,16],si8>
  return %result_q : !torch.vtensor<[3,2,4,16],si8>
}
