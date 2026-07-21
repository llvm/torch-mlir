// RUN: torch-mlir-opt -convert-torch-to-linalg -split-input-file %s | FileCheck %s

// Standalone dequantize — no matmul; lowered to linalg.generic by ConvertElementwiseOp.
// Checks are strict: SSA values are captured and threaded through each op so
// that a swap of a non-commutative operand (e.g. `arith.subi %zp, %ext` instead
// of `arith.subi %ext, %zp`) would fail the test.
//
// CHECK-LABEL: func.func @standalone_dequantize_si8(
// CHECK-SAME:    %[[ARG0:.*]]: !torch.vtensor<[4,8],si8>
// CHECK:       %[[IN_T:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[4,8],si8> -> tensor<4x8xi8>
// CHECK:       %[[SCALE_F:.*]] = torch.constant.float 3.000000e-01
// CHECK:       %[[OUT_INIT:.*]] = tensor.empty() : tensor<4x8xf32>
// CHECK:       %[[GEN:.*]] = linalg.generic
// CHECK-SAME:    ins(%[[IN_T]] : tensor<4x8xi8>) outs(%[[OUT_INIT]] : tensor<4x8xf32>)
// CHECK:       ^bb0(%[[IN:.*]]: i8, %{{.*}}: f32):
// CHECK:         %[[EXT:.*]] = arith.extsi %[[IN]] : i8 to i32
// CHECK:         %[[ZP_I64:.*]] = arith.constant 0 : i64
// CHECK:         %[[ZP_I32:.*]] = arith.trunci %[[ZP_I64]] : i64 to i32
// Operand order matters here: (input - zp), not (zp - input).
// CHECK:         %[[SUB:.*]] = arith.subi %[[EXT]], %[[ZP_I32]] : i32
// CHECK:         %[[SUBF:.*]] = arith.sitofp %[[SUB]] : i32 to f32
// CHECK:         %[[SCALE_F64:.*]] = torch_c.to_f64 %[[SCALE_F]]
// CHECK:         %[[SCALE_F32:.*]] = arith.truncf %[[SCALE_F64]] : f64 to f32
// CHECK:         %[[MUL:.*]] = arith.mulf %[[SUBF]], %[[SCALE_F32]] : f32
// CHECK:         linalg.yield %[[MUL]] : f32
func.func @standalone_dequantize_si8(
    %input: !torch.vtensor<[4,8],si8>) -> !torch.vtensor<[4,8],f32> {
  %scale = torch.constant.float 3.000000e-01
  %zp    = torch.constant.int 0
  %qmin  = torch.constant.int -128
  %qmax  = torch.constant.int 127
  %dtype = torch.constant.int 2
  %none  = torch.constant.none
  %od    = torch.derefine %none : !torch.none to !torch.optional<int>
  %out = torch.quantized_decomposed.dequantize_per_tensor
      %input, %scale, %zp, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[4,8],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
      -> !torch.vtensor<[4,8],f32>
  return %out : !torch.vtensor<[4,8],f32>
}

// -----

// Standalone quantize — no matmul; lowered to linalg.generic by ConvertElementwiseOp.
// Checks capture SSA values so a swap of e.g. `arith.divf %scale, %in` or
// `arith.maximumf %qmax, %v` (both non-commutative in effect) would fail.
//
// CHECK-LABEL: func.func @standalone_quantize_si8(
// CHECK-SAME:    %[[ARG0:.*]]: !torch.vtensor<[4,8],f32>
// CHECK:       %[[IN_T:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[4,8],f32> -> tensor<4x8xf32>
// CHECK:       %[[SCALE_F:.*]] = torch.constant.float 3.000000e-01
// CHECK:       %[[OUT_INIT:.*]] = tensor.empty() : tensor<4x8xi8>
// CHECK:       %[[GEN:.*]] = linalg.generic
// CHECK-SAME:    ins(%[[IN_T]] : tensor<4x8xf32>) outs(%[[OUT_INIT]] : tensor<4x8xi8>)
// CHECK:       ^bb0(%[[IN:.*]]: f32, %{{.*}}: i8):
// CHECK:         %[[QMIN_I64:.*]] = arith.constant -128 : i64
// CHECK:         %[[QMIN_F:.*]] = arith.sitofp %[[QMIN_I64]] : i64 to f32
// CHECK:         %[[QMAX_I64:.*]] = arith.constant 127 : i64
// CHECK:         %[[QMAX_F:.*]] = arith.sitofp %[[QMAX_I64]] : i64 to f32
// CHECK:         %[[SCALE_F64:.*]] = torch_c.to_f64 %[[SCALE_F]]
// CHECK:         %[[SCALE_F32:.*]] = arith.truncf %[[SCALE_F64]] : f64 to f32
// CHECK:         %[[ZP_I64:.*]] = arith.constant 0 : i64
// CHECK:         %[[ZP_F:.*]] = arith.sitofp %[[ZP_I64]] : i64 to f32
// Operand order matters here: (input / scale), not (scale / input).
// CHECK:         %[[DIV:.*]] = arith.divf %[[IN]], %[[SCALE_F32]] : f32
// CHECK:         %[[RND:.*]] = math.roundeven %[[DIV]] : f32
// CHECK:         %[[ADD:.*]] = arith.addf %[[RND]], %[[ZP_F]] : f32
// clamp: max(v, qmin) then min(., qmax); operand order fixes which side clamps.
// CHECK:         %[[CLAMP_LO:.*]] = arith.maximumf %[[ADD]], %[[QMIN_F]] : f32
// CHECK:         %[[CLAMP_HI:.*]] = arith.minimumf %[[CLAMP_LO]], %[[QMAX_F]] : f32
// CHECK:         %[[Q:.*]] = arith.fptosi %[[CLAMP_HI]] : f32 to i8
// CHECK:         linalg.yield %[[Q]] : i8
func.func @standalone_quantize_si8(
    %input: !torch.vtensor<[4,8],f32>) -> !torch.vtensor<[4,8],si8> {
  %scale = torch.constant.float 3.000000e-01
  %zp    = torch.constant.int 0
  %qmin  = torch.constant.int -128
  %qmax  = torch.constant.int 127
  %dtype = torch.constant.int 2
  %out = torch.quantized_decomposed.quantize_per_tensor
      %input, %scale, %zp, %qmin, %qmax, %dtype
      : !torch.vtensor<[4,8],f32>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int
      -> !torch.vtensor<[4,8],si8>
  return %out : !torch.vtensor<[4,8],si8>
}
