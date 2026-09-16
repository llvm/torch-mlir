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
// CHECK:         %[[ZP_I64:.*]] = arith.constant 0 : i64
// CHECK:         %[[EXT:.*]] = arith.extsi %[[IN]] : i8 to i64
// Operand order matters here: (input - zp), not (zp - input).
// CHECK:         %[[SUB:.*]] = arith.subi %[[EXT]], %[[ZP_I64]] : i64
// CHECK:         %[[SCALE_F64:.*]] = torch_c.to_f64 %[[SCALE_F]]
// CHECK:         %[[SCALE_F32:.*]] = arith.truncf %[[SCALE_F64]] : f64 to f32
// CHECK:         %[[SUBF:.*]] = arith.sitofp %[[SUB]] : i64 to f32
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

// -----

// CHECK-LABEL: func.func @quantize_per_tensor_tensor(
// CHECK-SAME:    %[[INPUT:[^:,]+]]: !torch.vtensor<[4,8],f32>,
// CHECK-SAME:    %[[SCALE:[^:,]+]]: !torch.vtensor<[],f32>,
// CHECK-SAME:    %[[ZP:[^:,]+]]: !torch.vtensor<[],si32>
// CHECK-DAG:   %[[ZP_T:.*]] = torch_c.to_builtin_tensor %[[ZP]] : !torch.vtensor<[],si32> -> tensor<i32>
// CHECK-DAG:   %[[SCALE_T:.*]] = torch_c.to_builtin_tensor %[[SCALE]] : !torch.vtensor<[],f32> -> tensor<f32>
// CHECK-DAG:   %[[IN_T:.*]] = torch_c.to_builtin_tensor %[[INPUT]] : !torch.vtensor<[4,8],f32> -> tensor<4x8xf32>
// CHECK-DAG:   %[[QMIN_I64:.*]] = arith.constant -128 : i64
// CHECK-DAG:   %[[QMAX_I64:.*]] = arith.constant 127 : i64
// CHECK-DAG:   %[[SCALE_V:.*]] = tensor.extract %[[SCALE_T]][] : tensor<f32>
// CHECK-DAG:   %[[ZP_V:.*]] = tensor.extract %[[ZP_T]][] : tensor<i32>
// CHECK-DAG:   %[[QMIN_F:.*]] = arith.sitofp %[[QMIN_I64]] : i64 to f32
// CHECK-DAG:   %[[QMAX_F:.*]] = arith.sitofp %[[QMAX_I64]] : i64 to f32
// CHECK:       %[[EMPTY:.*]] = tensor.empty() : tensor<4x8xi8>
// CHECK:       linalg.generic
// CHECK-SAME:    ins(%[[IN_T]] : tensor<4x8xf32>) outs(%[[EMPTY]] : tensor<4x8xi8>)
// CHECK:       ^bb0(%[[IN:.*]]: f32, %{{.*}}: i8):
// CHECK:         %[[ZP_F:.*]] = arith.sitofp %[[ZP_V]] : i32 to f32
// CHECK:         %[[DIV:.*]] = arith.divf %[[IN]], %[[SCALE_V]] : f32
// CHECK:         %[[RND:.*]] = math.roundeven %[[DIV]] : f32
// CHECK:         %[[ADD:.*]] = arith.addf %[[RND]], %[[ZP_F]] : f32
// CHECK:         %[[LO:.*]] = arith.maximumf %[[ADD]], %[[QMIN_F]] : f32
// CHECK:         %[[HI:.*]] = arith.minimumf %[[LO]], %[[QMAX_F]] : f32
// CHECK:         %[[Q:.*]] = arith.fptosi %[[HI]] : f32 to i8
// CHECK:         linalg.yield %[[Q]] : i8
func.func @quantize_per_tensor_tensor(
    %input: !torch.vtensor<[4,8],f32>,
    %scale: !torch.vtensor<[],f32>,
    %zp: !torch.vtensor<[],si32>)
    -> !torch.vtensor<[4,8],si8> {
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 2
  %out = torch.quantized_decomposed.quantize_per_tensor.tensor
      %input, %scale, %zp, %qmin, %qmax, %dtype
      : !torch.vtensor<[4,8],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],si32>,
        !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,8],si8>
  return %out : !torch.vtensor<[4,8],si8>
}

// -----

// CHECK-LABEL: func.func @quantize_per_tensor_tensor2(
// CHECK-SAME:    %[[INPUT:[^:,]+]]: !torch.vtensor<[4,8],f32>,
// CHECK-SAME:    %[[SCALE:[^:,]+]]: !torch.vtensor<[],f32>,
// CHECK-SAME:    %[[ZP:[^:,]+]]: !torch.vtensor<[],si32>,
// CHECK-SAME:    %[[QMIN:[^:,]+]]: !torch.vtensor<[],si32>,
// CHECK-SAME:    %[[QMAX:[^:,]+]]: !torch.vtensor<[],si32>
// CHECK-DAG:   %[[QMAX_T:.*]] = torch_c.to_builtin_tensor %[[QMAX]] : !torch.vtensor<[],si32> -> tensor<i32>
// CHECK-DAG:   %[[QMIN_T:.*]] = torch_c.to_builtin_tensor %[[QMIN]] : !torch.vtensor<[],si32> -> tensor<i32>
// CHECK-DAG:   %[[ZP_T:.*]] = torch_c.to_builtin_tensor %[[ZP]] : !torch.vtensor<[],si32> -> tensor<i32>
// CHECK-DAG:   %[[SCALE_T:.*]] = torch_c.to_builtin_tensor %[[SCALE]] : !torch.vtensor<[],f32> -> tensor<f32>
// CHECK-DAG:   %[[IN_T:.*]] = torch_c.to_builtin_tensor %[[INPUT]] : !torch.vtensor<[4,8],f32> -> tensor<4x8xf32>
// CHECK-DAG:   %[[SCALE_V:.*]] = tensor.extract %[[SCALE_T]][] : tensor<f32>
// CHECK-DAG:   %[[ZP_V:.*]] = tensor.extract %[[ZP_T]][] : tensor<i32>
// CHECK-DAG:   %[[QMIN_V:.*]] = tensor.extract %[[QMIN_T]][] : tensor<i32>
// CHECK-DAG:   %[[QMAX_V:.*]] = tensor.extract %[[QMAX_T]][] : tensor<i32>
// CHECK-DAG:   %[[QMIN_F:.*]] = arith.sitofp %[[QMIN_V]] : i32 to f32
// CHECK-DAG:   %[[QMAX_F:.*]] = arith.sitofp %[[QMAX_V]] : i32 to f32
// CHECK:       %[[EMPTY:.*]] = tensor.empty() : tensor<4x8xi8>
// CHECK:       linalg.generic
// CHECK-SAME:    ins(%[[IN_T]] : tensor<4x8xf32>) outs(%[[EMPTY]] : tensor<4x8xi8>)
// CHECK:       ^bb0(%[[IN:.*]]: f32, %{{.*}}: i8):
// CHECK:         %[[ZP_F:.*]] = arith.sitofp %[[ZP_V]] : i32 to f32
// CHECK:         %[[DIV:.*]] = arith.divf %[[IN]], %[[SCALE_V]] : f32
// CHECK:         %[[RND:.*]] = math.roundeven %[[DIV]] : f32
// CHECK:         %[[ADD:.*]] = arith.addf %[[RND]], %[[ZP_F]] : f32
// CHECK:         %[[LO:.*]] = arith.maximumf %[[ADD]], %[[QMIN_F]] : f32
// CHECK:         %[[HI:.*]] = arith.minimumf %[[LO]], %[[QMAX_F]] : f32
// CHECK:         %[[Q:.*]] = arith.fptosi %[[HI]] : f32 to i8
// CHECK:         linalg.yield %[[Q]] : i8
func.func @quantize_per_tensor_tensor2(
    %input: !torch.vtensor<[4,8],f32>,
    %scale: !torch.vtensor<[],f32>,
    %zp: !torch.vtensor<[],si32>,
    %qmin_t: !torch.vtensor<[],si32>,
    %qmax_t: !torch.vtensor<[],si32>)
    -> !torch.vtensor<[4,8],si8> {
  %dtype = torch.constant.int 2
  %out = torch.quantized_decomposed.quantize_per_tensor.tensor2
      %input, %scale, %zp, %qmin_t, %qmax_t, %dtype
      : !torch.vtensor<[4,8],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],si32>,
        !torch.vtensor<[],si32>, !torch.vtensor<[],si32>, !torch.int -> !torch.vtensor<[4,8],si8>
  return %out : !torch.vtensor<[4,8],si8>
}

// -----

// CHECK-LABEL: func.func @dequantize_per_tensor_tensor(
// CHECK-SAME:    %[[INPUT:[^:,]+]]: !torch.vtensor<[4,8],si8>,
// CHECK-SAME:    %[[SCALE:[^:,]+]]: !torch.vtensor<[],f32>,
// CHECK-SAME:    %[[ZP:[^:,]+]]: !torch.vtensor<[],si32>
// CHECK-DAG:   %[[SCALE_T:.*]] = torch_c.to_builtin_tensor %[[SCALE]] : !torch.vtensor<[],f32> -> tensor<f32>
// CHECK-DAG:   %[[ZP_T:.*]] = torch_c.to_builtin_tensor %[[ZP]] : !torch.vtensor<[],si32> -> tensor<i32>
// CHECK-DAG:   %[[SCALE_V:.*]] = tensor.extract %[[SCALE_T]][] : tensor<f32>
// CHECK-DAG:   %[[ZP_V:.*]] = tensor.extract %[[ZP_T]][] : tensor<i32>
// CHECK:       linalg.generic
// CHECK-SAME:    ins(%{{.*}} : tensor<4x8xi8>) outs(%{{.*}} : tensor<4x8xf32>)
// CHECK:       linalg.yield
func.func @dequantize_per_tensor_tensor(
    %input: !torch.vtensor<[4,8],si8>,
    %scale: !torch.vtensor<[],f32>,
    %zp: !torch.vtensor<[],si32>)
    -> !torch.vtensor<[4,8],f32> {
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 2
  %none = torch.constant.none
  %od = torch.derefine %none : !torch.none to !torch.optional<int>
  %out = torch.quantized_decomposed.dequantize_per_tensor.tensor
      %input, %scale, %zp, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[4,8],si8>, !torch.vtensor<[],f32>, !torch.vtensor<[],si32>,
        !torch.int, !torch.int, !torch.int, !torch.optional<int> -> !torch.vtensor<[4,8],f32>
  return %out : !torch.vtensor<[4,8],f32>
}

// -----

// CHECK-LABEL: func.func @dequantize_per_tensor_tensor2(
// CHECK-SAME:    %[[INPUT:[^:,]+]]: !torch.vtensor<[4,8],si8>,
// CHECK-SAME:    %[[SCALE:[^:,]+]]: !torch.vtensor<[],f32>,
// CHECK-SAME:    %[[ZP:[^:,]+]]: !torch.vtensor<[],si32>,
// CHECK-SAME:    %[[QMIN:[^:,]+]]: !torch.vtensor<[],si32>,
// CHECK-SAME:    %[[QMAX:[^:,]+]]: !torch.vtensor<[],si32>
// Note: qmin/qmax tensors are passed in but unused in the dequantize body.
// CHECK-DAG:   %[[ZP_T:.*]] = torch_c.to_builtin_tensor %[[ZP]] : !torch.vtensor<[],si32> -> tensor<i32>
// CHECK-DAG:   %[[SCALE_T:.*]] = torch_c.to_builtin_tensor %[[SCALE]] : !torch.vtensor<[],f32> -> tensor<f32>
// CHECK-DAG:   %[[IN_T:.*]] = torch_c.to_builtin_tensor %[[INPUT]] : !torch.vtensor<[4,8],si8> -> tensor<4x8xi8>
// CHECK-DAG:   %[[SCALE_V:.*]] = tensor.extract %[[SCALE_T]][] : tensor<f32>
// CHECK-DAG:   %[[ZP_V:.*]] = tensor.extract %[[ZP_T]][] : tensor<i32>
// CHECK:       %[[EMPTY:.*]] = tensor.empty() : tensor<4x8xf32>
// CHECK:       linalg.generic
// CHECK-SAME:    ins(%[[IN_T]] : tensor<4x8xi8>) outs(%[[EMPTY]] : tensor<4x8xf32>)
// CHECK:       ^bb0(%[[IN:.*]]: i8, %{{.*}}: f32):
// CHECK:         %[[EXT:.*]] = arith.extsi %[[IN]] : i8 to i32
// Operand order matters: (input - zp), not (zp - input).
// CHECK:         %[[SUB:.*]] = arith.subi %[[EXT]], %[[ZP_V]] : i32
// CHECK:         %[[FP:.*]] = arith.sitofp %[[SUB]] : i32 to f32
// CHECK:         %[[MUL:.*]] = arith.mulf %[[FP]], %[[SCALE_V]] : f32
// CHECK:         linalg.yield %[[MUL]] : f32
func.func @dequantize_per_tensor_tensor2(
    %input: !torch.vtensor<[4,8],si8>,
    %scale: !torch.vtensor<[],f32>,
    %zp: !torch.vtensor<[],si32>,
    %qmin_t: !torch.vtensor<[],si32>,
    %qmax_t: !torch.vtensor<[],si32>)
    -> !torch.vtensor<[4,8],f32> {
  %dtype = torch.constant.int 2
  %none = torch.constant.none
  %od = torch.derefine %none : !torch.none to !torch.optional<int>
  %out = torch.quantized_decomposed.dequantize_per_tensor.tensor2
      %input, %scale, %zp, %qmin_t, %qmax_t, %dtype, %od
      : !torch.vtensor<[4,8],si8>, !torch.vtensor<[],f32>, !torch.vtensor<[],si32>,
        !torch.vtensor<[],si32>, !torch.vtensor<[],si32>, !torch.int, !torch.optional<int>
        -> !torch.vtensor<[4,8],f32>
  return %out : !torch.vtensor<[4,8],f32>
}

// -----

// quantize_per_tensor.tensor with f16 input and unsigned ui8 output.
// CHECK-LABEL: func.func @quantize_per_tensor_tensor_f16_ui8(
// CHECK-SAME:    %[[INPUT:[^:,]+]]: !torch.vtensor<[4,8],f16>,
// CHECK-SAME:    %[[SCALE:[^:,]+]]: !torch.vtensor<[],f16>,
// CHECK-SAME:    %[[ZP:[^:,]+]]: !torch.vtensor<[],si32>
// CHECK-DAG:   %[[ZP_T:.*]] = torch_c.to_builtin_tensor %[[ZP]] : !torch.vtensor<[],si32> -> tensor<i32>
// CHECK-DAG:   %[[SCALE_T:.*]] = torch_c.to_builtin_tensor %[[SCALE]] : !torch.vtensor<[],f16> -> tensor<f16>
// CHECK-DAG:   %[[IN_T:.*]] = torch_c.to_builtin_tensor %[[INPUT]] : !torch.vtensor<[4,8],f16> -> tensor<4x8xf16>
// CHECK-DAG:   %[[QMIN_I64:.*]] = arith.constant 0 : i64
// CHECK-DAG:   %[[QMAX_I64:.*]] = arith.constant 255 : i64
// CHECK-DAG:   %[[SCALE_V:.*]] = tensor.extract %[[SCALE_T]][] : tensor<f16>
// CHECK-DAG:   %[[ZP_V:.*]] = tensor.extract %[[ZP_T]][] : tensor<i32>
// CHECK-DAG:   %[[QMIN_F:.*]] = arith.sitofp %[[QMIN_I64]] : i64 to f16
// CHECK-DAG:   %[[QMAX_F:.*]] = arith.sitofp %[[QMAX_I64]] : i64 to f16
// CHECK:       %[[EMPTY:.*]] = tensor.empty() : tensor<4x8xi8>
// CHECK:       linalg.generic
// CHECK-SAME:    ins(%[[IN_T]] : tensor<4x8xf16>) outs(%[[EMPTY]] : tensor<4x8xi8>)
// CHECK:       ^bb0(%[[IN:.*]]: f16, %{{.*}}: i8):
// CHECK:         %[[ZP_F:.*]] = arith.sitofp %[[ZP_V]] : i32 to f16
// CHECK:         %[[DIV:.*]] = arith.divf %[[IN]], %[[SCALE_V]] : f16
// CHECK:         %[[RND:.*]] = math.roundeven %[[DIV]] : f16
// CHECK:         %[[ADD:.*]] = arith.addf %[[RND]], %[[ZP_F]] : f16
// CHECK:         %[[LO:.*]] = arith.maximumf %[[ADD]], %[[QMIN_F]] : f16
// CHECK:         %[[HI:.*]] = arith.minimumf %[[LO]], %[[QMAX_F]] : f16
// Unsigned output path uses fptoui.
// CHECK:         %[[Q:.*]] = arith.fptoui %[[HI]] : f16 to i8
// CHECK:         linalg.yield %[[Q]] : i8
func.func @quantize_per_tensor_tensor_f16_ui8(
    %input: !torch.vtensor<[4,8],f16>,
    %scale: !torch.vtensor<[],f16>,
    %zp: !torch.vtensor<[],si32>)
    -> !torch.vtensor<[4,8],ui8> {
  %qmin = torch.constant.int 0
  %qmax = torch.constant.int 255
  %dtype = torch.constant.int 0
  %out = torch.quantized_decomposed.quantize_per_tensor.tensor
      %input, %scale, %zp, %qmin, %qmax, %dtype
      : !torch.vtensor<[4,8],f16>, !torch.vtensor<[],f16>, !torch.vtensor<[],si32>,
        !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,8],ui8>
  return %out : !torch.vtensor<[4,8],ui8>
}

// -----

// quantize_per_tensor.tensor2 with f16 input, ui8 output, tensor bounds.
// CHECK-LABEL: func.func @quantize_per_tensor_tensor2_f16_ui8(
// CHECK-SAME:    %[[INPUT:[^:,]+]]: !torch.vtensor<[2,4],f16>,
// CHECK-SAME:    %[[SCALE:[^:,]+]]: !torch.vtensor<[],f16>,
// CHECK-SAME:    %[[ZP:[^:,]+]]: !torch.vtensor<[],si32>,
// CHECK-SAME:    %[[QMIN:[^:,]+]]: !torch.vtensor<[],si32>,
// CHECK-SAME:    %[[QMAX:[^:,]+]]: !torch.vtensor<[],si32>
// CHECK-DAG:   %[[QMAX_T:.*]] = torch_c.to_builtin_tensor %[[QMAX]] : !torch.vtensor<[],si32> -> tensor<i32>
// CHECK-DAG:   %[[QMIN_T:.*]] = torch_c.to_builtin_tensor %[[QMIN]] : !torch.vtensor<[],si32> -> tensor<i32>
// CHECK-DAG:   %[[ZP_T:.*]] = torch_c.to_builtin_tensor %[[ZP]] : !torch.vtensor<[],si32> -> tensor<i32>
// CHECK-DAG:   %[[SCALE_T:.*]] = torch_c.to_builtin_tensor %[[SCALE]] : !torch.vtensor<[],f16> -> tensor<f16>
// CHECK-DAG:   %[[IN_T:.*]] = torch_c.to_builtin_tensor %[[INPUT]] : !torch.vtensor<[2,4],f16> -> tensor<2x4xf16>
// CHECK-DAG:   %[[SCALE_V:.*]] = tensor.extract %[[SCALE_T]][] : tensor<f16>
// CHECK-DAG:   %[[ZP_V:.*]] = tensor.extract %[[ZP_T]][] : tensor<i32>
// CHECK-DAG:   %[[QMIN_V:.*]] = tensor.extract %[[QMIN_T]][] : tensor<i32>
// CHECK-DAG:   %[[QMAX_V:.*]] = tensor.extract %[[QMAX_T]][] : tensor<i32>
// CHECK-DAG:   %[[QMIN_F:.*]] = arith.sitofp %[[QMIN_V]] : i32 to f16
// CHECK-DAG:   %[[QMAX_F:.*]] = arith.sitofp %[[QMAX_V]] : i32 to f16
// CHECK:       %[[EMPTY:.*]] = tensor.empty() : tensor<2x4xi8>
// CHECK:       linalg.generic
// CHECK-SAME:    ins(%[[IN_T]] : tensor<2x4xf16>) outs(%[[EMPTY]] : tensor<2x4xi8>)
// CHECK:       ^bb0(%[[IN:.*]]: f16, %{{.*}}: i8):
// CHECK:         %[[ZP_F:.*]] = arith.sitofp %[[ZP_V]] : i32 to f16
// CHECK:         %[[DIV:.*]] = arith.divf %[[IN]], %[[SCALE_V]] : f16
// CHECK:         %[[RND:.*]] = math.roundeven %[[DIV]] : f16
// CHECK:         %[[ADD:.*]] = arith.addf %[[RND]], %[[ZP_F]] : f16
// CHECK:         %[[LO:.*]] = arith.maximumf %[[ADD]], %[[QMIN_F]] : f16
// CHECK:         %[[HI:.*]] = arith.minimumf %[[LO]], %[[QMAX_F]] : f16
// CHECK:         %[[Q:.*]] = arith.fptoui %[[HI]] : f16 to i8
// CHECK:         linalg.yield %[[Q]] : i8
func.func @quantize_per_tensor_tensor2_f16_ui8(
    %input: !torch.vtensor<[2,4],f16>,
    %scale: !torch.vtensor<[],f16>,
    %zp: !torch.vtensor<[],si32>,
    %qmin_t: !torch.vtensor<[],si32>,
    %qmax_t: !torch.vtensor<[],si32>)
    -> !torch.vtensor<[2,4],ui8> {
  %dtype = torch.constant.int 0
  %out = torch.quantized_decomposed.quantize_per_tensor.tensor2
      %input, %scale, %zp, %qmin_t, %qmax_t, %dtype
      : !torch.vtensor<[2,4],f16>, !torch.vtensor<[],f16>, !torch.vtensor<[],si32>,
        !torch.vtensor<[],si32>, !torch.vtensor<[],si32>, !torch.int -> !torch.vtensor<[2,4],ui8>
  return %out : !torch.vtensor<[2,4],ui8>
}

// -----

// dequantize_per_tensor.tensor with unsigned ui8 input -> f32 output.
// CHECK-LABEL: func.func @dequantize_per_tensor_tensor_ui8(
// CHECK-SAME:    %[[INPUT:[^:,]+]]: !torch.vtensor<[4,8],ui8>,
// CHECK-SAME:    %[[SCALE:[^:,]+]]: !torch.vtensor<[],f32>,
// CHECK-SAME:    %[[ZP:[^:,]+]]: !torch.vtensor<[],si32>
// CHECK-DAG:   %[[ZP_T:.*]] = torch_c.to_builtin_tensor %[[ZP]] : !torch.vtensor<[],si32> -> tensor<i32>
// CHECK-DAG:   %[[SCALE_T:.*]] = torch_c.to_builtin_tensor %[[SCALE]] : !torch.vtensor<[],f32> -> tensor<f32>
// CHECK-DAG:   %[[IN_T:.*]] = torch_c.to_builtin_tensor %[[INPUT]] : !torch.vtensor<[4,8],ui8> -> tensor<4x8xi8>
// CHECK-DAG:   %[[SCALE_V:.*]] = tensor.extract %[[SCALE_T]][] : tensor<f32>
// CHECK-DAG:   %[[ZP_V:.*]] = tensor.extract %[[ZP_T]][] : tensor<i32>
// CHECK:       %[[EMPTY:.*]] = tensor.empty() : tensor<4x8xf32>
// CHECK:       linalg.generic
// CHECK-SAME:    ins(%[[IN_T]] : tensor<4x8xi8>) outs(%[[EMPTY]] : tensor<4x8xf32>)
// CHECK:       ^bb0(%[[IN:.*]]: i8, %{{.*}}: f32):
// Unsigned path: extui (not extsi) then (input - zp), then sitofp.
// CHECK:         %[[EXT:.*]] = arith.extui %[[IN]] : i8 to i32
// CHECK:         %[[SUB:.*]] = arith.subi %[[EXT]], %[[ZP_V]] : i32
// CHECK:         %[[FP:.*]] = arith.sitofp %[[SUB]] : i32 to f32
// CHECK:         %[[MUL:.*]] = arith.mulf %[[FP]], %[[SCALE_V]] : f32
// CHECK:         linalg.yield %[[MUL]] : f32
func.func @dequantize_per_tensor_tensor_ui8(
    %input: !torch.vtensor<[4,8],ui8>,
    %scale: !torch.vtensor<[],f32>,
    %zp: !torch.vtensor<[],si32>)
    -> !torch.vtensor<[4,8],f32> {
  %qmin = torch.constant.int 0
  %qmax = torch.constant.int 255
  %dtype = torch.constant.int 0
  %none = torch.constant.none
  %od = torch.derefine %none : !torch.none to !torch.optional<int>
  %out = torch.quantized_decomposed.dequantize_per_tensor.tensor
      %input, %scale, %zp, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[4,8],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],si32>,
        !torch.int, !torch.int, !torch.int, !torch.optional<int> -> !torch.vtensor<[4,8],f32>
  return %out : !torch.vtensor<[4,8],f32>
}

// -----

// dequantize_per_tensor.tensor2 with unsigned ui8 input, tensor bounds, f32 out.
// CHECK-LABEL: func.func @dequantize_per_tensor_tensor2_ui8(
// CHECK-SAME:    %[[INPUT:[^:,]+]]: !torch.vtensor<[2,4],ui8>,
// CHECK-SAME:    %[[SCALE:[^:,]+]]: !torch.vtensor<[],f32>,
// CHECK-SAME:    %[[ZP:[^:,]+]]: !torch.vtensor<[],si32>,
// CHECK-SAME:    %[[QMIN:[^:,]+]]: !torch.vtensor<[],si32>,
// CHECK-SAME:    %[[QMAX:[^:,]+]]: !torch.vtensor<[],si32>
// CHECK-DAG:   %[[ZP_T:.*]] = torch_c.to_builtin_tensor %[[ZP]] : !torch.vtensor<[],si32> -> tensor<i32>
// CHECK-DAG:   %[[SCALE_T:.*]] = torch_c.to_builtin_tensor %[[SCALE]] : !torch.vtensor<[],f32> -> tensor<f32>
// CHECK-DAG:   %[[IN_T:.*]] = torch_c.to_builtin_tensor %[[INPUT]] : !torch.vtensor<[2,4],ui8> -> tensor<2x4xi8>
// CHECK-DAG:   %[[SCALE_V:.*]] = tensor.extract %[[SCALE_T]][] : tensor<f32>
// CHECK-DAG:   %[[ZP_V:.*]] = tensor.extract %[[ZP_T]][] : tensor<i32>
// CHECK:       %[[EMPTY:.*]] = tensor.empty() : tensor<2x4xf32>
// CHECK:       linalg.generic
// CHECK-SAME:    ins(%[[IN_T]] : tensor<2x4xi8>) outs(%[[EMPTY]] : tensor<2x4xf32>)
// CHECK:       ^bb0(%[[IN:.*]]: i8, %{{.*}}: f32):
// CHECK:         %[[EXT:.*]] = arith.extui %[[IN]] : i8 to i32
// CHECK:         %[[SUB:.*]] = arith.subi %[[EXT]], %[[ZP_V]] : i32
// CHECK:         %[[FP:.*]] = arith.sitofp %[[SUB]] : i32 to f32
// CHECK:         %[[MUL:.*]] = arith.mulf %[[FP]], %[[SCALE_V]] : f32
// CHECK:         linalg.yield %[[MUL]] : f32
func.func @dequantize_per_tensor_tensor2_ui8(
    %input: !torch.vtensor<[2,4],ui8>,
    %scale: !torch.vtensor<[],f32>,
    %zp: !torch.vtensor<[],si32>,
    %qmin_t: !torch.vtensor<[],si32>,
    %qmax_t: !torch.vtensor<[],si32>)
    -> !torch.vtensor<[2,4],f32> {
  %dtype = torch.constant.int 0
  %none = torch.constant.none
  %od = torch.derefine %none : !torch.none to !torch.optional<int>
  %out = torch.quantized_decomposed.dequantize_per_tensor.tensor2
      %input, %scale, %zp, %qmin_t, %qmax_t, %dtype, %od
      : !torch.vtensor<[2,4],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],si32>,
        !torch.vtensor<[],si32>, !torch.vtensor<[],si32>, !torch.int, !torch.optional<int>
        -> !torch.vtensor<[2,4],f32>
  return %out : !torch.vtensor<[2,4],f32>
}

// -----

// choose_qparams.tensor: asymmetric per-tensor calibration, f32 input, si8 range.
// CHECK-LABEL: func.func @choose_qparams_tensor(
// CHECK-SAME:    %[[INPUT:[^:,]+]]: !torch.vtensor<[4,8],f32>
// CHECK:       %[[IN_T:.*]] = torch_c.to_builtin_tensor %[[INPUT]] : !torch.vtensor<[4,8],f32> -> tensor<4x8xf32>
// CHECK:       %[[COLL:.*]] = tensor.collapse_shape %[[IN_T]]
// CHECK:       %[[NEG_INF:.*]] = arith.constant 0xFF800000 : f32
// CHECK:       %[[POS_INF:.*]] = arith.constant 0x7F800000 : f32
// CHECK:       %[[INIT_MIN:.*]] = tensor.from_elements %[[POS_INF]] : tensor<f32>
// CHECK:       %[[INIT_MAX:.*]] = tensor.from_elements %[[NEG_INF]] : tensor<f32>
// CHECK:       %[[REDUCE:.*]]:2 = linalg.generic
// CHECK-SAME:    ins(%[[COLL]] : tensor<32xf32>) outs(%[[INIT_MIN]], %[[INIT_MAX]] : tensor<f32>, tensor<f32>)
// CHECK:       ^bb0(%[[R_IN:.*]]: f32, %[[R_MIN:.*]]: f32, %[[R_MAX:.*]]: f32):
// CHECK:         %[[NEW_MIN:.*]] = arith.minimumf %[[R_IN]], %[[R_MIN]] : f32
// CHECK:         %[[NEW_MAX:.*]] = arith.maximumf %[[R_IN]], %[[R_MAX]] : f32
// CHECK:         linalg.yield %[[NEW_MIN]], %[[NEW_MAX]] : f32, f32
// CHECK:       %[[QMIN_F:.*]] = arith.constant -1.280000e+02 : f32
// CHECK:       %[[QMAX_F:.*]] = arith.constant 1.270000e+02 : f32
// CHECK:       %[[MIN_V:.*]] = tensor.extract %[[REDUCE]]#0[] : tensor<f32>
// CHECK:       %[[MAX_V:.*]] = tensor.extract %[[REDUCE]]#1[] : tensor<f32>
// CHECK:       %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:       %[[MIN_C:.*]] = arith.minimumf %[[MIN_V]], %[[ZERO]] : f32
// CHECK:       %[[MAX_C:.*]] = arith.maximumf %[[MAX_V]], %[[ZERO]] : f32
// CHECK:       %[[QRANGE:.*]] = arith.subf %[[QMAX_F]], %[[QMIN_F]] : f32
// CHECK:       %[[DRANGE:.*]] = arith.subf %[[MAX_C]], %[[MIN_C]] : f32
// CHECK:       %[[RAW_SCALE:.*]] = arith.divf %[[DRANGE]], %[[QRANGE]] : f32
// CHECK:       %[[SCALE_EPS:.*]] = arith.maximumf %[[RAW_SCALE]], %{{.*}} : f32
// CHECK:       %[[MIN_OVER_SCALE:.*]] = arith.divf %[[MIN_C]], %[[SCALE_EPS]] : f32
// CHECK:       %[[RND:.*]] = math.roundeven %[[MIN_OVER_SCALE]] : f32
// CHECK:       %[[RAW_ZP:.*]] = arith.subf %[[QMIN_F]], %[[RND]] : f32
// CHECK:       %[[ZP_LO:.*]] = arith.maximumf %[[RAW_ZP]], %[[QMIN_F]] : f32
// CHECK:       %[[ZP_HI:.*]] = arith.minimumf %[[ZP_LO]], %[[QMAX_F]] : f32
// CHECK:       %[[ZP_I64:.*]] = arith.fptosi %[[ZP_HI]] : f32 to i64
// CHECK:       %[[DEG:.*]] = arith.cmpf ogt, %[[MIN_V]], %[[MAX_V]] : f32
// CHECK:       %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:       %[[ZERO_I64:.*]] = arith.constant 0 : i64
// CHECK:       %[[FINAL_SCALE:.*]] = arith.select %[[DEG]], %[[ONE]], %[[SCALE_EPS]] : f32
// CHECK:       %[[FINAL_ZP:.*]] = arith.select %[[DEG]], %[[ZERO_I64]], %[[ZP_I64]] : i64
// CHECK:       tensor.from_elements %[[FINAL_SCALE]] : tensor<1xf32>
// CHECK:       tensor.from_elements %[[FINAL_ZP]] : tensor<1xi64>
func.func @choose_qparams_tensor(
    %input: !torch.vtensor<[4,8],f32>)
    -> (!torch.vtensor<[1],f32>, !torch.vtensor<[1],si64>) {
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %eps = torch.constant.float 1.000000e-08
  %dtype = torch.constant.int 2
  %scale, %zp = torch.quantized_decomposed.choose_qparams.tensor
      %input, %qmin, %qmax, %eps, %dtype
      : !torch.vtensor<[4,8],f32>, !torch.int, !torch.int, !torch.float, !torch.int
      -> !torch.vtensor<[1],f32>, !torch.vtensor<[1],si64>
  return %scale, %zp : !torch.vtensor<[1],f32>, !torch.vtensor<[1],si64>
}

// -----

// Symmetric round-trip: choose_qparams_symmetric.tensor -> quantize_per_tensor.tensor2
// CHECK-LABEL: func.func @choose_qparams_symmetric_quantize2_dequantize2_roundtrip(
// CHECK-SAME:    %[[INPUT:[^:,]+]]: !torch.vtensor<[?,?],f32>,
// CHECK-SAME:    %[[QMIN:[^:,]+]]: !torch.vtensor<[],si32>,
// CHECK-SAME:    %[[QMAX:[^:,]+]]: !torch.vtensor<[],si32>
// CHECK-DAG:   %[[QMAX_T:.*]] = torch_c.to_builtin_tensor %[[QMAX]] : !torch.vtensor<[],si32> -> tensor<i32>
// CHECK-DAG:   %[[QMIN_T:.*]] = torch_c.to_builtin_tensor %[[QMIN]] : !torch.vtensor<[],si32> -> tensor<i32>
// CHECK-DAG:   %[[IN_T:.*]] = torch_c.to_builtin_tensor %[[INPUT]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:       %[[COLL:.*]] = tensor.collapse_shape %[[IN_T]]
// CHECK:       %[[REDUCE:.*]]:2 = linalg.generic
// CHECK-SAME:    ins(%[[COLL]] : tensor<?xf32>)
// CHECK:       tensor.extract %[[REDUCE]]#0[] : tensor<f32>
// CHECK:       tensor.extract %[[REDUCE]]#1[] : tensor<f32>
// CHECK:       %[[C0_I64:.*]] = arith.constant 0 : i64
// CHECK:       %[[SCALE_1D:.*]] = tensor.from_elements %{{.*}} : tensor<1xf32>
// CHECK:       %[[ZP_1D:.*]] = tensor.from_elements %{{.*}} : tensor<1xi64>
// CHECK:       %[[SCALE_Q:.*]] = tensor.extract %[[SCALE_1D]][%{{.*}}] : tensor<1xf32>
// CHECK:       %[[ZP_Q:.*]] = tensor.extract %[[ZP_1D]][%{{.*}}] : tensor<1xi64>
// CHECK-DAG:   %[[QMIN_V:.*]] = tensor.extract %[[QMIN_T]][] : tensor<i32>
// CHECK-DAG:   %[[QMAX_V:.*]] = tensor.extract %[[QMAX_T]][] : tensor<i32>
// Quantize generic lowers the f32 input to si8.
// CHECK:       %[[Q_GEN:.*]] = linalg.generic
// CHECK-SAME:    ins(%[[IN_T]] : tensor<?x?xf32>)
// CHECK-SAME:    outs(%{{.*}} : tensor<?x?xi8>)
// CHECK:       linalg.yield %{{.*}} : i8
// CHECK:       %[[SCALE_DQ:.*]] = tensor.extract %[[SCALE_1D]][%{{.*}}] : tensor<1xf32>
// CHECK:       %[[ZP_DQ:.*]] = tensor.extract %[[ZP_1D]][%{{.*}}] : tensor<1xi64>
// CHECK:       linalg.generic
// CHECK-SAME:    ins(%[[Q_GEN]] : tensor<?x?xi8>)
// CHECK-SAME:    outs(%{{.*}} : tensor<?x?xf32>)
// CHECK:       linalg.yield %{{.*}} : f32
func.func @choose_qparams_symmetric_quantize2_dequantize2_roundtrip(
    %input:  !torch.vtensor<[?,?],f32>,
    %qmin_t: !torch.vtensor<[],si32>,
    %qmax_t: !torch.vtensor<[],si32>)
    -> !torch.vtensor<[?,?],f32> {
  %qmin  = torch.constant.int -128
  %qmax  = torch.constant.int 127
  %eps   = torch.constant.float 1.000000e-08
  %dtype = torch.constant.int 2
  %scale, %zp = torch.quantized_decomposed.choose_qparams_symmetric.tensor
      %input, %qmin, %qmax, %eps, %dtype
      : !torch.vtensor<[?,?],f32>, !torch.int, !torch.int, !torch.float, !torch.int
      -> !torch.vtensor<[1],f32>, !torch.vtensor<[1],si64>
  %quantized = torch.quantized_decomposed.quantize_per_tensor.tensor2
      %input, %scale, %zp, %qmin_t, %qmax_t, %dtype
      : !torch.vtensor<[?,?],f32>, !torch.vtensor<[1],f32>, !torch.vtensor<[1],si64>,
        !torch.vtensor<[],si32>, !torch.vtensor<[],si32>, !torch.int
        -> !torch.vtensor<[?,?],si8>
  %none  = torch.constant.none
  %od    = torch.derefine %none : !torch.none to !torch.optional<int>
  %out   = torch.quantized_decomposed.dequantize_per_tensor.tensor2
      %quantized, %scale, %zp, %qmin_t, %qmax_t, %dtype, %od
      : !torch.vtensor<[?,?],si8>, !torch.vtensor<[1],f32>, !torch.vtensor<[1],si64>,
        !torch.vtensor<[],si32>, !torch.vtensor<[],si32>, !torch.int, !torch.optional<int>
        -> !torch.vtensor<[?,?],f32>
  return %out : !torch.vtensor<[?,?],f32>
}

// -----

// Round-trip: choose_qparams.tensor -> quantize_per_tensor.tensor -> dequantize_per_tensor.tensor
// with dynamic dimensions
// CHECK-LABEL: func.func @choose_qparams_quantize_dequantize_roundtrip(
// CHECK-SAME:    %[[INPUT:[^:,]+]]: !torch.vtensor<[?,?],f32>
// CHECK:       %[[IN_T:.*]] = torch_c.to_builtin_tensor %[[INPUT]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK-DAG:   %[[QMIN_I64:.*]] = arith.constant -128 : i64
// CHECK-DAG:   %[[QMAX_I64:.*]] = arith.constant 127 : i64
// CHECK:       %[[COLL:.*]] = tensor.collapse_shape %[[IN_T]]
// CHECK:       %[[REDUCE:.*]]:2 = linalg.generic
// CHECK-SAME:    ins(%[[COLL]] : tensor<?xf32>)
// CHECK:       tensor.extract %[[REDUCE]]#0[] : tensor<f32>
// CHECK:       tensor.extract %[[REDUCE]]#1[] : tensor<f32>
// CHECK:       %[[SCALE_1D:.*]] = tensor.from_elements %{{.*}} : tensor<1xf32>
// CHECK:       %[[ZP_1D:.*]] = tensor.from_elements %{{.*}} : tensor<1xi64>
// CHECK:       %[[SCALE_Q:.*]] = tensor.extract %[[SCALE_1D]][%{{.*}}] : tensor<1xf32>
// CHECK:       %[[ZP_Q:.*]] = tensor.extract %[[ZP_1D]][%{{.*}}] : tensor<1xi64>
// CHECK:       arith.sitofp %[[QMIN_I64]] : i64 to f32
// CHECK:       arith.sitofp %[[QMAX_I64]] : i64 to f32
// CHECK:       %[[Q_GEN:.*]] = linalg.generic
// CHECK-SAME:    ins(%[[IN_T]] : tensor<?x?xf32>)
// CHECK-SAME:    outs(%{{.*}} : tensor<?x?xi8>)
// CHECK:       linalg.yield %{{.*}} : i8
// CHECK:       %[[SCALE_DQ:.*]] = tensor.extract %[[SCALE_1D]][%{{.*}}] : tensor<1xf32>
// CHECK:       %[[ZP_DQ:.*]] = tensor.extract %[[ZP_1D]][%{{.*}}] : tensor<1xi64>
// CHECK:       linalg.generic
// CHECK-SAME:    ins(%[[Q_GEN]] : tensor<?x?xi8>)
// CHECK-SAME:    outs(%{{.*}} : tensor<?x?xf32>)
// CHECK:       linalg.yield %{{.*}} : f32
func.func @choose_qparams_quantize_dequantize_roundtrip(
    %input: !torch.vtensor<[?,?],f32>)
    -> !torch.vtensor<[?,?],f32> {
  %qmin  = torch.constant.int -128
  %qmax  = torch.constant.int 127
  %eps   = torch.constant.float 1.000000e-08
  %dtype = torch.constant.int 2
  %scale, %zp = torch.quantized_decomposed.choose_qparams.tensor
      %input, %qmin, %qmax, %eps, %dtype
      : !torch.vtensor<[?,?],f32>, !torch.int, !torch.int, !torch.float, !torch.int
      -> !torch.vtensor<[1],f32>, !torch.vtensor<[1],si64>
  %quantized = torch.quantized_decomposed.quantize_per_tensor.tensor
      %input, %scale, %zp, %qmin, %qmax, %dtype
      : !torch.vtensor<[?,?],f32>, !torch.vtensor<[1],f32>, !torch.vtensor<[1],si64>,
        !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,?],si8>
  %none  = torch.constant.none
  %od    = torch.derefine %none : !torch.none to !torch.optional<int>
  %out   = torch.quantized_decomposed.dequantize_per_tensor.tensor
      %quantized, %scale, %zp, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[?,?],si8>, !torch.vtensor<[1],f32>, !torch.vtensor<[1],si64>,
        !torch.int, !torch.int, !torch.int, !torch.optional<int>
        -> !torch.vtensor<[?,?],f32>
  return %out : !torch.vtensor<[?,?],f32>
}

// -----

// choose_qparams_symmetric.tensor: symmetric per-tensor calibration, f32 input, si8 range.
// CHECK-LABEL: func.func @choose_qparams_symmetric_tensor(
// CHECK-SAME:    %[[INPUT:[^:,]+]]: !torch.vtensor<[4,8],f32>
// CHECK:       %[[IN_T:.*]] = torch_c.to_builtin_tensor %[[INPUT]] : !torch.vtensor<[4,8],f32> -> tensor<4x8xf32>
// CHECK:       %[[COLL:.*]] = tensor.collapse_shape %[[IN_T]]
// CHECK:       %[[REDUCE:.*]]:2 = linalg.generic
// CHECK-SAME:    ins(%[[COLL]] : tensor<32xf32>)
// CHECK:       %[[QMIN_F:.*]] = arith.constant -1.280000e+02 : f32
// CHECK:       %[[QMAX_F:.*]] = arith.constant 1.270000e+02 : f32
// CHECK:       %[[MIN_V:.*]] = tensor.extract %[[REDUCE]]#0[] : tensor<f32>
// CHECK:       %[[MAX_V:.*]] = tensor.extract %[[REDUCE]]#1[] : tensor<f32>
// CHECK:       %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:       %[[MIN_C:.*]] = arith.minimumf %[[MIN_V]], %[[ZERO]] : f32
// CHECK:       %[[MAX_C:.*]] = arith.maximumf %[[MAX_V]], %[[ZERO]] : f32
// CHECK:       %[[QRANGE:.*]] = arith.subf %[[QMAX_F]], %[[QMIN_F]] : f32
// CHECK:       %[[NEG_MIN:.*]] = arith.negf %[[MIN_C]] : f32
// CHECK:       %[[ABSMAX:.*]] = arith.maximumf %[[NEG_MIN]], %[[MAX_C]] : f32
// CHECK:       %[[TWO:.*]] = arith.constant 2.000000e+00 : f32
// CHECK:       %[[HALF_RANGE:.*]] = arith.divf %[[QRANGE]], %[[TWO]] : f32
// CHECK:       %[[RAW_SCALE:.*]] = arith.divf %[[ABSMAX]], %[[HALF_RANGE]] : f32
// CHECK:       %[[SCALE_EPS:.*]] = arith.maximumf %[[RAW_SCALE]], %{{.*}} : f32
// CHECK:       %[[ZP_I64:.*]] = arith.constant 0 : i64
// CHECK:       %[[DEG:.*]] = arith.cmpf ogt, %[[MIN_V]], %[[MAX_V]] : f32
// CHECK:       %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:       %[[ZERO_I64:.*]] = arith.constant 0 : i64
// CHECK:       %[[FINAL_SCALE:.*]] = arith.select %[[DEG]], %[[ONE]], %[[SCALE_EPS]] : f32
// CHECK:       %[[FINAL_ZP:.*]] = arith.select %[[DEG]], %[[ZERO_I64]], %[[ZP_I64]] : i64
// CHECK:       tensor.from_elements %[[FINAL_SCALE]] : tensor<1xf32>
// CHECK:       tensor.from_elements %[[FINAL_ZP]] : tensor<1xi64>
func.func @choose_qparams_symmetric_tensor(
    %input: !torch.vtensor<[4,8],f32>)
    -> (!torch.vtensor<[1],f32>, !torch.vtensor<[1],si64>) {
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %eps = torch.constant.float 1.000000e-08
  %dtype = torch.constant.int 2
  %scale, %zp = torch.quantized_decomposed.choose_qparams_symmetric.tensor
      %input, %qmin, %qmax, %eps, %dtype
      : !torch.vtensor<[4,8],f32>, !torch.int, !torch.int, !torch.float, !torch.int
      -> !torch.vtensor<[1],f32>, !torch.vtensor<[1],si64>
  return %scale, %zp : !torch.vtensor<[1],f32>, !torch.vtensor<[1],si64>
}

// -----

// CHECK: #[[IDENTITY:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[CHANNEL0:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL: func.func @dequantize_per_channel_axis0(
// CHECK-SAME: %[[INPUT:.*]]: !torch.vtensor<[4,8],si8>
// CHECK-SAME: %[[SCALES:.*]]: !torch.vtensor<[4],f32>
// CHECK-SAME: %[[ZPS:.*]]: !torch.vtensor<[4],si64>
// CHECK-DAG: %[[INPUT_T:.*]] = torch_c.to_builtin_tensor %[[INPUT]]
// CHECK-DAG: %[[SCALES_T:.*]] = torch_c.to_builtin_tensor %[[SCALES]]
// CHECK-DAG: %[[ZPS_T:.*]] = torch_c.to_builtin_tensor %[[ZPS]]
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<4x8xf32>
// CHECK: %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[IDENTITY]], #[[CHANNEL0]], #[[CHANNEL0]], #[[IDENTITY]]]
// CHECK-SAME: ins(%[[INPUT_T]], %[[SCALES_T]], %[[ZPS_T]] : tensor<4x8xi8>, tensor<4xf32>, tensor<4xi64>)
// CHECK-SAME: outs(%[[EMPTY]] : tensor<4x8xf32>)
// CHECK: ^bb0(%[[IN:.*]]: i8, %[[SCALE:.*]]: f32, %[[ZP:.*]]: i64, %{{.*}}: f32):
// CHECK:   %[[EXT:.*]] = arith.extsi %[[IN]] : i8 to i64
// CHECK:   %[[SUB:.*]] = arith.subi %[[EXT]], %[[ZP]] : i64
// CHECK:   %[[FP:.*]] = arith.sitofp %[[SUB]] : i64 to f32
// CHECK:   %[[MUL:.*]] = arith.mulf %[[FP]], %[[SCALE]] : f32
// CHECK:   linalg.yield %[[MUL]] : f32
func.func @dequantize_per_channel_axis0(
    %input: !torch.vtensor<[4,8],si8>,
    %scales: !torch.vtensor<[4],f32>,
    %zero_points: !torch.vtensor<[4],si64>)
    -> !torch.vtensor<[4,8],f32> {
  %axis = torch.constant.int 0
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 2
  %none = torch.constant.none
  %od = torch.derefine %none : !torch.none to !torch.optional<int>
  %out = torch.quantized_decomposed.dequantize_per_channel
      %input, %scales, %zero_points, %axis, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[4,8],si8>, !torch.vtensor<[4],f32>,
        !torch.vtensor<[4],si64>, !torch.int, !torch.int, !torch.int,
        !torch.int, !torch.optional<int> -> !torch.vtensor<[4,8],f32>
  return %out : !torch.vtensor<[4,8],f32>
}

// -----

// CHECK: #[[IDENTITY:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[CHANNEL1:.*]] = affine_map<(d0, d1) -> (d1)>
// CHECK-LABEL: func.func @dequantize_per_channel_same_width(
// CHECK: %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[IDENTITY]], #[[CHANNEL1]], #[[CHANNEL1]], #[[IDENTITY]]]
// CHECK-SAME: ins({{.*}} : tensor<3x4xi8>, tensor<4xf32>, tensor<4xi8>)
// CHECK: ^bb0(%[[IN:.*]]: i8, %[[SCALE:.*]]: f32, %[[ZP:.*]]: i8, %{{.*}}: f32):
// CHECK-DAG: %[[EXT_IN:.*]] = arith.extsi %[[IN]] : i8 to i16
// CHECK-DAG: %[[EXT_ZP:.*]] = arith.extsi %[[ZP]] : i8 to i16
// CHECK:   %[[SUB:.*]] = arith.subi %[[EXT_IN]], %[[EXT_ZP]] : i16
// CHECK:   %[[FP:.*]] = arith.sitofp %[[SUB]] : i16 to f32
// CHECK:   %[[MUL:.*]] = arith.mulf %[[FP]], %[[SCALE]] : f32
// CHECK:   linalg.yield %[[MUL]] : f32
func.func @dequantize_per_channel_same_width(
    %input: !torch.vtensor<[3,4],si8>,
    %scales: !torch.vtensor<[4],f32>,
    %zero_points: !torch.vtensor<[4],si8>)
    -> !torch.vtensor<[3,4],f32> {
  %axis = torch.constant.int 1
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 2
  %none = torch.constant.none
  %od = torch.derefine %none : !torch.none to !torch.optional<int>
  %out = torch.quantized_decomposed.dequantize_per_channel
      %input, %scales, %zero_points, %axis, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[3,4],si8>, !torch.vtensor<[4],f32>,
        !torch.vtensor<[4],si8>, !torch.int, !torch.int, !torch.int,
        !torch.int, !torch.optional<int> -> !torch.vtensor<[3,4],f32>
  return %out : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: func.func @dequantize_per_channel_unsigned_zero_point(
// CHECK: ^bb0(%[[IN:.*]]: i8, %{{.*}}: f32, %[[ZP:.*]]: i8, %{{.*}}: f32):
// CHECK:   arith.extui %[[ZP]] : i8 to i16
// CHECK:   arith.extui %[[IN]] : i8 to i16
func.func @dequantize_per_channel_unsigned_zero_point(
    %input: !torch.vtensor<[3,4],ui8>,
    %scales: !torch.vtensor<[4],f32>,
    %zero_points: !torch.vtensor<[4],ui8>)
    -> !torch.vtensor<[3,4],f32> {
  %axis = torch.constant.int 1
  %qmin = torch.constant.int 0
  %qmax = torch.constant.int 255
  %dtype = torch.constant.int 0
  %none = torch.constant.none
  %od = torch.derefine %none : !torch.none to !torch.optional<int>
  %out = torch.quantized_decomposed.dequantize_per_channel
      %input, %scales, %zero_points, %axis, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[3,4],ui8>, !torch.vtensor<[4],f32>,
        !torch.vtensor<[4],ui8>, !torch.int, !torch.int, !torch.int,
        !torch.int, !torch.optional<int> -> !torch.vtensor<[3,4],f32>
  return %out : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: func.func @dequantize_per_channel_mixed_signedness(
// CHECK: ^bb0(%[[IN:.*]]: i8, %{{.*}}: f32, %[[ZP:.*]]: i8, %{{.*}}: f32):
// CHECK:   arith.extui %[[ZP]] : i8 to i16
// CHECK:   arith.extsi %[[IN]] : i8 to i16
func.func @dequantize_per_channel_mixed_signedness(
    %input: !torch.vtensor<[3,4],si8>,
    %scales: !torch.vtensor<[4],f32>,
    %zero_points: !torch.vtensor<[4],ui8>)
    -> !torch.vtensor<[3,4],f32> {
  %axis = torch.constant.int 1
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 2
  %none = torch.constant.none
  %od = torch.derefine %none : !torch.none to !torch.optional<int>
  %out = torch.quantized_decomposed.dequantize_per_channel
      %input, %scales, %zero_points, %axis, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[3,4],si8>, !torch.vtensor<[4],f32>,
        !torch.vtensor<[4],ui8>, !torch.int, !torch.int, !torch.int,
        !torch.int, !torch.optional<int> -> !torch.vtensor<[3,4],f32>
  return %out : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK: #[[IDENTITY:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[CHANNEL0:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL: func.func @dequantize_per_channel_symmetric(
// CHECK: %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[IDENTITY]], #[[CHANNEL0]], #[[IDENTITY]]]
// CHECK-SAME: ins({{.*}} : tensor<4x8xi8>, tensor<4xf32>)
// CHECK: ^bb0(%[[IN:.*]]: i8, %[[SCALE:.*]]: f32, %{{.*}}: f32):
// CHECK-NOT: arith.subi
// CHECK:   %[[FP:.*]] = arith.sitofp %[[IN]] : i8 to f32
// CHECK:   %[[MUL:.*]] = arith.mulf %[[FP]], %[[SCALE]] : f32
// CHECK:   linalg.yield %[[MUL]] : f32
func.func @dequantize_per_channel_symmetric(
    %input: !torch.vtensor<[4,8],si8>,
    %scales: !torch.vtensor<[4],f32>)
    -> !torch.vtensor<[4,8],f32> {
  %axis = torch.constant.int 0
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 2
  %none = torch.constant.none
  %od = torch.derefine %none : !torch.none to !torch.optional<int>
  %out = torch.quantized_decomposed.dequantize_per_channel
      %input, %scales, %none, %axis, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[4,8],si8>, !torch.vtensor<[4],f32>,
        !torch.none, !torch.int, !torch.int, !torch.int,
        !torch.int, !torch.optional<int> -> !torch.vtensor<[4,8],f32>
  return %out : !torch.vtensor<[4,8],f32>
}

// -----

// CHECK: #[[IDENTITY:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[CHANNEL1:.*]] = affine_map<(d0, d1) -> (d1)>
// CHECK-LABEL: func.func @quantize_per_channel_axis1(
// CHECK-SAME: %[[INPUT:.*]]: !torch.vtensor<[4,8],f32>
// CHECK-SAME: %[[SCALES:.*]]: !torch.vtensor<[8],f32>
// CHECK-SAME: %[[ZPS:.*]]: !torch.vtensor<[8],si64>
// CHECK-DAG: %[[INPUT_T:.*]] = torch_c.to_builtin_tensor %[[INPUT]]
// CHECK-DAG: %[[SCALES_T:.*]] = torch_c.to_builtin_tensor %[[SCALES]]
// CHECK-DAG: %[[ZPS_T:.*]] = torch_c.to_builtin_tensor %[[ZPS]]
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<4x8xi8>
// CHECK: %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[IDENTITY]], #[[CHANNEL1]], #[[CHANNEL1]], #[[IDENTITY]]]
// CHECK-SAME: ins(%[[INPUT_T]], %[[SCALES_T]], %[[ZPS_T]] : tensor<4x8xf32>, tensor<8xf32>, tensor<8xi64>)
// CHECK-SAME: outs(%[[EMPTY]] : tensor<4x8xi8>)
// CHECK: ^bb0(%[[IN:.*]]: f32, %[[SCALE:.*]]: f32, %[[ZP:.*]]: i64, %{{.*}}: i8):
// CHECK:   %[[QMIN:.*]] = arith.constant -1.280000e+02 : f32
// CHECK:   %[[QMAX:.*]] = arith.constant 1.270000e+02 : f32
// CHECK:   %[[ZPF:.*]] = arith.sitofp %[[ZP]] : i64 to f32
// CHECK:   %[[DIV:.*]] = arith.divf %[[IN]], %[[SCALE]] : f32
// CHECK:   %[[ROUND:.*]] = math.roundeven %[[DIV]] : f32
// CHECK:   %[[ADD:.*]] = arith.addf %[[ROUND]], %[[ZPF]] : f32
// CHECK:   %[[LOW:.*]] = arith.maximumf %[[ADD]], %[[QMIN]] : f32
// CHECK:   %[[HIGH:.*]] = arith.minimumf %[[LOW]], %[[QMAX]] : f32
// CHECK:   %[[RESULT:.*]] = arith.fptosi %[[HIGH]] : f32 to i8
// CHECK:   linalg.yield %[[RESULT]] : i8
func.func @quantize_per_channel_axis1(
    %input: !torch.vtensor<[4,8],f32>,
    %scales: !torch.vtensor<[8],f32>,
    %zero_points: !torch.vtensor<[8],si64>)
    -> !torch.vtensor<[4,8],si8> {
  %axis = torch.constant.int 1
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 2
  %out = torch.quantized_decomposed.quantize_per_channel
      %input, %scales, %zero_points, %axis, %qmin, %qmax, %dtype
      : !torch.vtensor<[4,8],f32>, !torch.vtensor<[8],f32>,
        !torch.vtensor<[8],si64>, !torch.int, !torch.int, !torch.int,
        !torch.int -> !torch.vtensor<[4,8],si8>
  return %out : !torch.vtensor<[4,8],si8>
}

// -----

// CHECK-LABEL: func.func @quantize_per_channel_unsigned_zero_point(
// CHECK: ^bb0(%{{.*}}: f32, %{{.*}}: f32, %[[ZP:.*]]: i8, %{{.*}}: i8):
// CHECK:   arith.uitofp %[[ZP]] : i8 to f32
func.func @quantize_per_channel_unsigned_zero_point(
    %input: !torch.vtensor<[4,8],f32>,
    %scales: !torch.vtensor<[8],f32>,
    %zero_points: !torch.vtensor<[8],ui8>)
    -> !torch.vtensor<[4,8],ui8> {
  %axis = torch.constant.int 1
  %qmin = torch.constant.int 0
  %qmax = torch.constant.int 255
  %dtype = torch.constant.int 0
  %out = torch.quantized_decomposed.quantize_per_channel
      %input, %scales, %zero_points, %axis, %qmin, %qmax, %dtype
      : !torch.vtensor<[4,8],f32>, !torch.vtensor<[8],f32>,
        !torch.vtensor<[8],ui8>, !torch.int, !torch.int, !torch.int,
        !torch.int -> !torch.vtensor<[4,8],ui8>
  return %out : !torch.vtensor<[4,8],ui8>
}

// -----

// The multiply uses the scale type and only its result is narrowed to the
// explicit out_dtype.
// CHECK-LABEL: func.func @dequantize_per_channel_scale_truncation(
// CHECK: %[[FP:.*]] = arith.sitofp %{{.*}} : i8 to f32
// CHECK: %[[MUL:.*]] = arith.mulf %[[FP]], %{{.*}} : f32
// CHECK: arith.truncf %[[MUL]] : f32 to f16
func.func @dequantize_per_channel_scale_truncation(
    %input: !torch.vtensor<[4,8],si8>,
    %scales: !torch.vtensor<[4],f32>)
    -> !torch.vtensor<[4,8],f16> {
  %axis = torch.constant.int 0
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 2
  %out_dtype = torch.constant.int 5
  %none = torch.constant.none
  %out = torch.quantized_decomposed.dequantize_per_channel
      %input, %scales, %none, %axis, %qmin, %qmax, %dtype, %out_dtype
      : !torch.vtensor<[4,8],si8>, !torch.vtensor<[4],f32>,
        !torch.none, !torch.int, !torch.int, !torch.int,
        !torch.int, !torch.int -> !torch.vtensor<[4,8],f16>
  return %out : !torch.vtensor<[4,8],f16>
}

// -----

// The multiply uses the scale type and only its result is extended to the
// explicit out_dtype.
// CHECK-LABEL: func.func @dequantize_per_channel_scale_extension(
// CHECK: %[[FP:.*]] = arith.sitofp %{{.*}} : i8 to f16
// CHECK: %[[MUL:.*]] = arith.mulf %[[FP]], %{{.*}} : f16
// CHECK: arith.extf %[[MUL]] : f16 to f32
func.func @dequantize_per_channel_scale_extension(
    %input: !torch.vtensor<[4,8],si8>,
    %scales: !torch.vtensor<[4],f16>)
    -> !torch.vtensor<[4,8],f32> {
  %axis = torch.constant.int 0
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 2
  %out_dtype = torch.constant.int 6
  %none = torch.constant.none
  %out = torch.quantized_decomposed.dequantize_per_channel
      %input, %scales, %none, %axis, %qmin, %qmax, %dtype, %out_dtype
      : !torch.vtensor<[4,8],si8>, !torch.vtensor<[4],f16>,
        !torch.none, !torch.int, !torch.int, !torch.int,
        !torch.int, !torch.int -> !torch.vtensor<[4,8],f32>
  return %out : !torch.vtensor<[4,8],f32>
}

// -----

// Check only the unsigned branches of the shared quantize/dequantize payloads.
// CHECK-LABEL: func.func @per_channel_unsigned(
// CHECK: arith.fptoui %{{.*}} : f32 to i8
// CHECK: arith.extui %{{.*}} : i8 to i64
// CHECK: arith.sitofp %{{.*}} : i64 to f32
func.func @per_channel_unsigned(
    %input: !torch.vtensor<[4,8],f32>,
    %scales: !torch.vtensor<[4],f32>,
    %zero_points: !torch.vtensor<[4],si64>)
    -> !torch.vtensor<[4,8],f32> {
  %axis = torch.constant.int 0
  %qmin = torch.constant.int 0
  %qmax = torch.constant.int 255
  %dtype = torch.constant.int 0
  %none = torch.constant.none
  %od = torch.derefine %none : !torch.none to !torch.optional<int>
  %quantized = torch.quantized_decomposed.quantize_per_channel
      %input, %scales, %zero_points, %axis, %qmin, %qmax, %dtype
      : !torch.vtensor<[4,8],f32>, !torch.vtensor<[4],f32>,
        !torch.vtensor<[4],si64>, !torch.int, !torch.int, !torch.int,
        !torch.int -> !torch.vtensor<[4,8],ui8>
  %out = torch.quantized_decomposed.dequantize_per_channel
      %quantized, %scales, %zero_points, %axis, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[4,8],ui8>, !torch.vtensor<[4],f32>,
        !torch.vtensor<[4],si64>, !torch.int, !torch.int, !torch.int,
        !torch.int, !torch.optional<int> -> !torch.vtensor<[4,8],f32>
  return %out : !torch.vtensor<[4,8],f32>
}

// -----

// Without a zero point, an unsigned input retains its unsigned interpretation
// through the conversion to the scale type.
// CHECK-LABEL: func.func @dequantize_per_channel_unsigned_symmetric(
// CHECK-NOT: arith.extui
// CHECK-NOT: arith.subi
// CHECK: arith.uitofp %{{.*}} : i8 to f32
func.func @dequantize_per_channel_unsigned_symmetric(
    %input: !torch.vtensor<[4,8],ui8>,
    %scales: !torch.vtensor<[4],f32>)
    -> !torch.vtensor<[4,8],f32> {
  %axis = torch.constant.int 0
  %qmin = torch.constant.int 0
  %qmax = torch.constant.int 255
  %dtype = torch.constant.int 0
  %none = torch.constant.none
  %od = torch.derefine %none : !torch.none to !torch.optional<int>
  %out = torch.quantized_decomposed.dequantize_per_channel
      %input, %scales, %none, %axis, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[4,8],ui8>, !torch.vtensor<[4],f32>,
        !torch.none, !torch.int, !torch.int, !torch.int,
        !torch.int, !torch.optional<int> -> !torch.vtensor<[4,8],f32>
  return %out : !torch.vtensor<[4,8],f32>
}

// -----

// A negative axis is normalized before constructing the channel map.
// CHECK: #[[IDENTITY:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[LAST_DIM:.*]] = affine_map<(d0, d1) -> (d1)>
// CHECK-LABEL: func.func @dequantize_per_channel_negative_axis(
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[IDENTITY]], #[[LAST_DIM]], #[[IDENTITY]]]
func.func @dequantize_per_channel_negative_axis(
    %input: !torch.vtensor<[4,8],si8>,
    %scales: !torch.vtensor<[8],f32>)
    -> !torch.vtensor<[4,8],f32> {
  %axis = torch.constant.int -1
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 2
  %none = torch.constant.none
  %od = torch.derefine %none : !torch.none to !torch.optional<int>
  %out = torch.quantized_decomposed.dequantize_per_channel
      %input, %scales, %none, %axis, %qmin, %qmax, %dtype, %od
      : !torch.vtensor<[4,8],si8>, !torch.vtensor<[8],f32>,
        !torch.none, !torch.int, !torch.int, !torch.int,
        !torch.int, !torch.optional<int> -> !torch.vtensor<[4,8],f32>
  return %out : !torch.vtensor<[4,8],f32>
}

// -----

// Per-channel-group dequantization.
// CHECK: #[[IDENTITY:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[GROUP:.*]] = affine_map<(d0, d1) -> (d0, d1 floordiv 4)>
// CHECK-LABEL: func.func @dequantize_per_channel_group(
// CHECK-SAME: %[[INPUT:.*]]: !torch.vtensor<[4,16],si8>
// CHECK-SAME: %[[SCALES:.*]]: !torch.vtensor<[4,4],f32>
// CHECK-SAME: %[[ZPS:.*]]: !torch.vtensor<[4,4],si64>
// CHECK-DAG: %[[INPUT_T:.*]] = torch_c.to_builtin_tensor %[[INPUT]]
// CHECK-DAG: %[[SCALES_T:.*]] = torch_c.to_builtin_tensor %[[SCALES]]
// CHECK-DAG: %[[ZPS_T:.*]] = torch_c.to_builtin_tensor %[[ZPS]]
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<4x16xf32>
// CHECK: %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[IDENTITY]], #[[GROUP]], #[[GROUP]], #[[IDENTITY]]]
// CHECK-SAME: ins(%[[INPUT_T]], %[[SCALES_T]], %[[ZPS_T]] : tensor<4x16xi8>, tensor<4x4xf32>, tensor<4x4xi64>)
// CHECK-SAME: outs(%[[EMPTY]] : tensor<4x16xf32>)
// CHECK: ^bb0(%[[IN:.*]]: i8, %[[SCALE:.*]]: f32, %[[ZP:.*]]: i64, %{{.*}}: f32):
// CHECK:   %[[EXT:.*]] = arith.extsi %[[IN]] : i8 to i64
// CHECK:   %[[SUB:.*]] = arith.subi %[[EXT]], %[[ZP]] : i64
// CHECK:   %[[FP:.*]] = arith.sitofp %[[SUB]] : i64 to f32
// CHECK:   %[[MUL:.*]] = arith.mulf %[[FP]], %[[SCALE]] : f32
// CHECK:   linalg.yield %[[MUL]] : f32
func.func @dequantize_per_channel_group(
    %input: !torch.vtensor<[4,16],si8>,
    %scales: !torch.vtensor<[4,4],f32>,
    %zero_points: !torch.vtensor<[4,4],si64>)
    -> !torch.vtensor<[4,16],f32> {
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 2
  %group_size = torch.constant.int 4
  %out_dtype = torch.constant.int 6
  %out = torch.quantized_decomposed.dequantize_per_channel_group
      %input, %scales, %zero_points, %qmin, %qmax, %dtype, %group_size, %out_dtype
      : !torch.vtensor<[4,16],si8>, !torch.vtensor<[4,4],f32>,
        !torch.vtensor<[4,4],si64>, !torch.int, !torch.int, !torch.int,
        !torch.int, !torch.int -> !torch.vtensor<[4,16],f32>
  return %out : !torch.vtensor<[4,16],f32>
}

// -----

// Per-channel-group symmetric dequantization (no zero points).
// CHECK: #[[IDENTITY:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[GROUP:.*]] = affine_map<(d0, d1) -> (d0, d1 floordiv 8)>
// CHECK-LABEL: func.func @dequantize_per_channel_group_symmetric(
// CHECK: %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[IDENTITY]], #[[GROUP]], #[[IDENTITY]]]
// CHECK-SAME: ins({{.*}} : tensor<8x32xi8>, tensor<8x4xf32>)
// CHECK: ^bb0(%[[IN:.*]]: i8, %[[SCALE:.*]]: f32, %{{.*}}: f32):
// CHECK-NOT: arith.subi
// CHECK:   %[[FP:.*]] = arith.sitofp %[[IN]] : i8 to f32
// CHECK:   %[[MUL:.*]] = arith.mulf %[[FP]], %[[SCALE]] : f32
// CHECK:   linalg.yield %[[MUL]] : f32
func.func @dequantize_per_channel_group_symmetric(
    %input: !torch.vtensor<[8,32],si8>,
    %scales: !torch.vtensor<[8,4],f32>)
    -> !torch.vtensor<[8,32],f32> {
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 2
  %group_size = torch.constant.int 8
  %out_dtype = torch.constant.int 6
  %none = torch.constant.none
  %out = torch.quantized_decomposed.dequantize_per_channel_group
      %input, %scales, %none, %qmin, %qmax, %dtype, %group_size, %out_dtype
      : !torch.vtensor<[8,32],si8>, !torch.vtensor<[8,4],f32>,
        !torch.none, !torch.int, !torch.int, !torch.int,
        !torch.int, !torch.int -> !torch.vtensor<[8,32],f32>
  return %out : !torch.vtensor<[8,32],f32>
}

// -----

// Per-channel-group quantization.
// CHECK: #[[IDENTITY:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[GROUP:.*]] = affine_map<(d0, d1) -> (d0, d1 floordiv 4)>
// CHECK-LABEL: func.func @quantize_per_channel_group(
// CHECK-SAME: %[[INPUT:.*]]: !torch.vtensor<[4,16],f32>
// CHECK-SAME: %[[SCALES:.*]]: !torch.vtensor<[4,4],f32>
// CHECK-SAME: %[[ZPS:.*]]: !torch.vtensor<[4,4],si64>
// CHECK-DAG: %[[INPUT_T:.*]] = torch_c.to_builtin_tensor %[[INPUT]]
// CHECK-DAG: %[[SCALES_T:.*]] = torch_c.to_builtin_tensor %[[SCALES]]
// CHECK-DAG: %[[ZPS_T:.*]] = torch_c.to_builtin_tensor %[[ZPS]]
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<4x16xi8>
// CHECK: %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[IDENTITY]], #[[GROUP]], #[[GROUP]], #[[IDENTITY]]]
// CHECK-SAME: ins(%[[INPUT_T]], %[[SCALES_T]], %[[ZPS_T]] : tensor<4x16xf32>, tensor<4x4xf32>, tensor<4x4xi64>)
// CHECK-SAME: outs(%[[EMPTY]] : tensor<4x16xi8>)
// CHECK: ^bb0(%[[IN:.*]]: f32, %[[SCALE:.*]]: f32, %[[ZP:.*]]: i64, %{{.*}}: i8):
// CHECK:   %[[QMIN:.*]] = arith.constant -1.280000e+02 : f32
// CHECK:   %[[QMAX:.*]] = arith.constant 1.270000e+02 : f32
// CHECK:   %[[ZPF:.*]] = arith.sitofp %[[ZP]] : i64 to f32
// CHECK:   %[[DIV:.*]] = arith.divf %[[IN]], %[[SCALE]] : f32
// CHECK:   %[[ROUND:.*]] = math.roundeven %[[DIV]] : f32
// CHECK:   %[[ADD:.*]] = arith.addf %[[ROUND]], %[[ZPF]] : f32
// CHECK:   %[[LOW:.*]] = arith.maximumf %[[ADD]], %[[QMIN]] : f32
// CHECK:   %[[HIGH:.*]] = arith.minimumf %[[LOW]], %[[QMAX]] : f32
// CHECK:   %[[RESULT:.*]] = arith.fptosi %[[HIGH]] : f32 to i8
// CHECK:   linalg.yield %[[RESULT]] : i8
func.func @quantize_per_channel_group(
    %input: !torch.vtensor<[4,16],f32>,
    %scales: !torch.vtensor<[4,4],f32>,
    %zero_points: !torch.vtensor<[4,4],si64>)
    -> !torch.vtensor<[4,16],si8> {
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 2
  %group_size = torch.constant.int 4
  %out = torch.quantized_decomposed.quantize_per_channel_group
      %input, %scales, %zero_points, %qmin, %qmax, %dtype, %group_size
      : !torch.vtensor<[4,16],f32>, !torch.vtensor<[4,4],f32>,
        !torch.vtensor<[4,4],si64>, !torch.int, !torch.int, !torch.int,
        !torch.int -> !torch.vtensor<[4,16],si8>
  return %out : !torch.vtensor<[4,16],si8>
}

// -----

// Per-channel-group with same-width zero points (i8 input, i8 zero_points).
// CHECK: #[[IDENTITY:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[GROUP:.*]] = affine_map<(d0, d1) -> (d0, d1 floordiv 4)>
// CHECK-LABEL: func.func @dequantize_per_channel_group_same_width_zp(
// CHECK: %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[IDENTITY]], #[[GROUP]], #[[GROUP]], #[[IDENTITY]]]
// CHECK-SAME: ins({{.*}} : tensor<4x16xi8>, tensor<4x4xf32>, tensor<4x4xi8>)
// CHECK: ^bb0(%[[IN:.*]]: i8, %[[SCALE:.*]]: f32, %[[ZP:.*]]: i8, %{{.*}}: f32):
// CHECK-DAG: %[[EXT_IN:.*]] = arith.extsi %[[IN]] : i8 to i16
// CHECK-DAG: %[[EXT_ZP:.*]] = arith.extsi %[[ZP]] : i8 to i16
// CHECK:   %[[SUB:.*]] = arith.subi %[[EXT_IN]], %[[EXT_ZP]] : i16
// CHECK:   %[[FP:.*]] = arith.sitofp %[[SUB]] : i16 to f32
// CHECK:   %[[MUL:.*]] = arith.mulf %[[FP]], %[[SCALE]] : f32
// CHECK:   linalg.yield %[[MUL]] : f32
func.func @dequantize_per_channel_group_same_width_zp(
    %input: !torch.vtensor<[4,16],si8>,
    %scales: !torch.vtensor<[4,4],f32>,
    %zero_points: !torch.vtensor<[4,4],si8>)
    -> !torch.vtensor<[4,16],f32> {
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 2
  %group_size = torch.constant.int 4
  %out_dtype = torch.constant.int 6
  %out = torch.quantized_decomposed.dequantize_per_channel_group
      %input, %scales, %zero_points, %qmin, %qmax, %dtype, %group_size, %out_dtype
      : !torch.vtensor<[4,16],si8>, !torch.vtensor<[4,4],f32>,
        !torch.vtensor<[4,4],si8>, !torch.int, !torch.int, !torch.int,
        !torch.int, !torch.int -> !torch.vtensor<[4,16],f32>
  return %out : !torch.vtensor<[4,16],f32>
}

// -----

// Per-channel-group unsigned quantization (uint8).
// CHECK-LABEL: func.func @per_channel_group_unsigned(
// CHECK: arith.fptoui %{{.*}} : f32 to i8
// CHECK: arith.extui %{{.*}} : i8 to i64
// CHECK: arith.sitofp %{{.*}} : i64 to f32
func.func @per_channel_group_unsigned(
    %input: !torch.vtensor<[4,16],f32>,
    %scales: !torch.vtensor<[4,4],f32>,
    %zero_points: !torch.vtensor<[4,4],si64>)
    -> !torch.vtensor<[4,16],f32> {
  %qmin = torch.constant.int 0
  %qmax = torch.constant.int 255
  %dtype = torch.constant.int 0
  %group_size = torch.constant.int 4
  %out_dtype = torch.constant.int 6
  %quantized = torch.quantized_decomposed.quantize_per_channel_group
      %input, %scales, %zero_points, %qmin, %qmax, %dtype, %group_size
      : !torch.vtensor<[4,16],f32>, !torch.vtensor<[4,4],f32>,
        !torch.vtensor<[4,4],si64>, !torch.int, !torch.int, !torch.int,
        !torch.int -> !torch.vtensor<[4,16],ui8>
  %out = torch.quantized_decomposed.dequantize_per_channel_group
      %quantized, %scales, %zero_points, %qmin, %qmax, %dtype, %group_size, %out_dtype
      : !torch.vtensor<[4,16],ui8>, !torch.vtensor<[4,4],f32>,
        !torch.vtensor<[4,4],si64>, !torch.int, !torch.int, !torch.int,
        !torch.int, !torch.int -> !torch.vtensor<[4,16],f32>
  return %out : !torch.vtensor<[4,16],f32>
}

// -----

// Per-channel-group unsigned symmetric dequantization uses uitofp directly.
// CHECK-LABEL: func.func @dequantize_per_channel_group_unsigned_symmetric(
// CHECK-NOT: arith.extui
// CHECK-NOT: arith.subi
// CHECK: arith.uitofp %{{.*}} : i8 to f32
func.func @dequantize_per_channel_group_unsigned_symmetric(
    %input: !torch.vtensor<[4,16],ui8>,
    %scales: !torch.vtensor<[4,4],f32>)
    -> !torch.vtensor<[4,16],f32> {
  %qmin = torch.constant.int 0
  %qmax = torch.constant.int 255
  %dtype = torch.constant.int 0
  %group_size = torch.constant.int 4
  %out_dtype = torch.constant.int 6
  %none = torch.constant.none
  %out = torch.quantized_decomposed.dequantize_per_channel_group
      %input, %scales, %none, %qmin, %qmax, %dtype, %group_size, %out_dtype
      : !torch.vtensor<[4,16],ui8>, !torch.vtensor<[4,4],f32>,
        !torch.none, !torch.int, !torch.int, !torch.int,
        !torch.int, !torch.int -> !torch.vtensor<[4,16],f32>
  return %out : !torch.vtensor<[4,16],f32>
}

// -----

// Per-channel-group with f16 scales and f32 output: multiply in f16, then extend.
// CHECK-LABEL: func.func @dequantize_per_channel_group_scale_extension(
// CHECK: %[[FP:.*]] = arith.sitofp %{{.*}} : i8 to f16
// CHECK: %[[MUL:.*]] = arith.mulf %[[FP]], %{{.*}} : f16
// CHECK: arith.extf %[[MUL]] : f16 to f32
func.func @dequantize_per_channel_group_scale_extension(
    %input: !torch.vtensor<[4,16],si8>,
    %scales: !torch.vtensor<[4,4],f16>)
    -> !torch.vtensor<[4,16],f32> {
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 2
  %group_size = torch.constant.int 4
  %out_dtype = torch.constant.int 6
  %none = torch.constant.none
  %out = torch.quantized_decomposed.dequantize_per_channel_group
      %input, %scales, %none, %qmin, %qmax, %dtype, %group_size, %out_dtype
      : !torch.vtensor<[4,16],si8>, !torch.vtensor<[4,4],f16>,
        !torch.none, !torch.int, !torch.int, !torch.int,
        !torch.int, !torch.int -> !torch.vtensor<[4,16],f32>
  return %out : !torch.vtensor<[4,16],f32>
}

// -----

// Per-channel-group with f32 scales and f16 output: multiply in f32, then truncate.
// CHECK-LABEL: func.func @dequantize_per_channel_group_scale_truncation(
// CHECK: %[[FP:.*]] = arith.sitofp %{{.*}} : i8 to f32
// CHECK: %[[MUL:.*]] = arith.mulf %[[FP]], %{{.*}} : f32
// CHECK: arith.truncf %[[MUL]] : f32 to f16
func.func @dequantize_per_channel_group_scale_truncation(
    %input: !torch.vtensor<[4,16],si8>,
    %scales: !torch.vtensor<[4,4],f32>)
    -> !torch.vtensor<[4,16],f16> {
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 2
  %group_size = torch.constant.int 4
  %out_dtype = torch.constant.int 5
  %none = torch.constant.none
  %out = torch.quantized_decomposed.dequantize_per_channel_group
      %input, %scales, %none, %qmin, %qmax, %dtype, %group_size, %out_dtype
      : !torch.vtensor<[4,16],si8>, !torch.vtensor<[4,4],f32>,
        !torch.none, !torch.int, !torch.int, !torch.int,
        !torch.int, !torch.int -> !torch.vtensor<[4,16],f16>
  return %out : !torch.vtensor<[4,16],f16>
}

// -----

// Per-channel-group with 4-bit quantization.
// CHECK: #[[IDENTITY:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[GROUP:.*]] = affine_map<(d0, d1) -> (d0, d1 floordiv 128)>
// CHECK-LABEL: func.func @dequantize_per_channel_group_int4(
// CHECK: %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[IDENTITY]], #[[GROUP]], #[[GROUP]], #[[IDENTITY]]]
// CHECK-SAME: ins({{.*}} : tensor<4096x4096xi8>, tensor<4096x32xf16>, tensor<4096x32xi8>)
// CHECK-SAME: outs({{.*}} : tensor<4096x4096xf16>)
func.func @dequantize_per_channel_group_int4(
    %input: !torch.vtensor<[4096,4096],si8>,
    %scales: !torch.vtensor<[4096,32],f16>,
    %zero_points: !torch.vtensor<[4096,32],si8>)
    -> !torch.vtensor<[4096,4096],f16> {
  %qmin = torch.constant.int -8
  %qmax = torch.constant.int 7
  %dtype = torch.constant.int 2
  %group_size = torch.constant.int 128
  %out_dtype = torch.constant.int 5
  %out = torch.quantized_decomposed.dequantize_per_channel_group
      %input, %scales, %zero_points, %qmin, %qmax, %dtype, %group_size, %out_dtype
      : !torch.vtensor<[4096,4096],si8>, !torch.vtensor<[4096,32],f16>,
        !torch.vtensor<[4096,32],si8>, !torch.int, !torch.int, !torch.int,
        !torch.int, !torch.int -> !torch.vtensor<[4096,4096],f16>
  return %out : !torch.vtensor<[4096,4096],f16>
}

// -----

// GPTQ single-column dequantize: group_size (128) > input.shape[-1] (16) and
// scales.shape[-1] == 1, so group_size is clamped to 16 before lowering.
// The affine map must use floordiv 16 (not floordiv 128).
// CHECK: #[[IDENTITY:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[GROUP:.*]] = affine_map<(d0, d1) -> (d0, d1 floordiv 16)>
// CHECK-LABEL: func.func @dequantize_per_channel_group_gptq_single_col(
// CHECK: %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[IDENTITY]], #[[GROUP]], #[[GROUP]], #[[IDENTITY]]]
// CHECK-SAME: ins({{.*}} : tensor<4x16xi8>, tensor<4x1xf32>, tensor<4x1xi64>)
// CHECK-SAME: outs({{.*}} : tensor<4x16xf32>)
func.func @dequantize_per_channel_group_gptq_single_col(
    %input: !torch.vtensor<[4,16],si8>,
    %scales: !torch.vtensor<[4,1],f32>,
    %zero_points: !torch.vtensor<[4,1],si64>)
    -> !torch.vtensor<[4,16],f32> {
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 2
  %group_size = torch.constant.int 128
  %out_dtype = torch.constant.int 6
  %out = torch.quantized_decomposed.dequantize_per_channel_group
      %input, %scales, %zero_points, %qmin, %qmax, %dtype, %group_size, %out_dtype
      : !torch.vtensor<[4,16],si8>, !torch.vtensor<[4,1],f32>,
        !torch.vtensor<[4,1],si64>, !torch.int, !torch.int, !torch.int,
        !torch.int, !torch.int -> !torch.vtensor<[4,16],f32>
  return %out : !torch.vtensor<[4,16],f32>
}

// -----

// GPTQ single-column quantize: group_size (128) > input.shape[-1] (16) and
// scales.shape[-1] == 1, so group_size is clamped to 16 before lowering.
// The affine map must use floordiv 16 (not floordiv 128).
// CHECK: #[[IDENTITY:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[GROUP:.*]] = affine_map<(d0, d1) -> (d0, d1 floordiv 16)>
// CHECK-LABEL: func.func @quantize_per_channel_group_gptq_single_col(
// CHECK: %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[IDENTITY]], #[[GROUP]], #[[GROUP]], #[[IDENTITY]]]
// CHECK-SAME: ins({{.*}} : tensor<4x16xf32>, tensor<4x1xf32>, tensor<4x1xi64>)
// CHECK-SAME: outs({{.*}} : tensor<4x16xi8>)
func.func @quantize_per_channel_group_gptq_single_col(
    %input: !torch.vtensor<[4,16],f32>,
    %scales: !torch.vtensor<[4,1],f32>,
    %zero_points: !torch.vtensor<[4,1],si64>)
    -> !torch.vtensor<[4,16],si8> {
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 2
  %group_size = torch.constant.int 128
  %out = torch.quantized_decomposed.quantize_per_channel_group
      %input, %scales, %zero_points, %qmin, %qmax, %dtype, %group_size
      : !torch.vtensor<[4,16],f32>, !torch.vtensor<[4,1],f32>,
        !torch.vtensor<[4,1],si64>, !torch.int, !torch.int, !torch.int,
        !torch.int -> !torch.vtensor<[4,16],si8>
  return %out : !torch.vtensor<[4,16],si8>
}

// -----

// Per-channel-group round trip on dynamic dimensions.
// CHECK: #[[IDENTITY:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[GROUP:.*]] = affine_map<(d0, d1) -> (d0, d1 floordiv 4)>
// CHECK-LABEL: func.func @per_channel_group_dynamic_roundtrip(
// CHECK: torch_c.to_builtin_tensor %{{.*}} : !torch.vtensor<[?,?],si64> -> tensor<?x?xi64>
// CHECK: tensor.empty({{.*}}) : tensor<?x?xi8>
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[IDENTITY]], #[[GROUP]], #[[GROUP]], #[[IDENTITY]]]
// CHECK-SAME: ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xi64>)
// CHECK-SAME: outs(%{{.*}} : tensor<?x?xi8>)
// CHECK: ^bb0(%[[IN:.*]]: f32, %[[SCALE:.*]]: f32, %[[ZP:.*]]: i64, %{{.*}}: i8):
// CHECK:   %[[QMIN:.*]] = arith.constant -1.280000e+02 : f32
// CHECK:   %[[QMAX:.*]] = arith.constant 1.270000e+02 : f32
// CHECK:   %[[ZPF:.*]] = arith.sitofp %[[ZP]] : i64 to f32
// CHECK:   %[[DIV:.*]] = arith.divf %[[IN]], %[[SCALE]] : f32
// CHECK:   %[[RND:.*]] = math.roundeven %[[DIV]] : f32
// CHECK:   %[[ADD:.*]] = arith.addf %[[RND]], %[[ZPF]] : f32
// CHECK:   %[[LOW:.*]] = arith.maximumf %[[ADD]], %[[QMIN]] : f32
// CHECK:   %[[HIGH:.*]] = arith.minimumf %[[LOW]], %[[QMAX]] : f32
// CHECK:   %[[QV:.*]] = arith.fptosi %[[HIGH]] : f32 to i8
// CHECK:   linalg.yield %[[QV]] : i8
// CHECK: tensor.empty({{.*}}) : tensor<?x?xf32>
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[IDENTITY]], #[[GROUP]], #[[GROUP]], #[[IDENTITY]]]
// CHECK-SAME: ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<?x?xi8>, tensor<?x?xf32>, tensor<?x?xi64>)
// CHECK-SAME: outs(%{{.*}} : tensor<?x?xf32>)
// CHECK: ^bb0(%[[QIN:.*]]: i8, %[[SC2:.*]]: f32, %[[ZP2:.*]]: i64, %{{.*}}: f32):
// CHECK:   %[[EXT:.*]] = arith.extsi %[[QIN]] : i8 to i64
// CHECK:   %[[SUB:.*]] = arith.subi %[[EXT]], %[[ZP2]] : i64
// CHECK:   %[[FP:.*]] = arith.sitofp %[[SUB]] : i64 to f32
// CHECK:   %[[MUL:.*]] = arith.mulf %[[FP]], %[[SC2]] : f32
// CHECK:   linalg.yield %[[MUL]] : f32
func.func @per_channel_group_dynamic_roundtrip(
    %input: !torch.vtensor<[?,?],f32>,
    %scales: !torch.vtensor<[?,?],f32>,
    %zero_points: !torch.vtensor<[?,?],si64>)
    -> !torch.vtensor<[?,?],f32> {
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 2
  %group_size = torch.constant.int 4
  %out_dtype = torch.constant.int 6
  %quantized = torch.quantized_decomposed.quantize_per_channel_group
      %input, %scales, %zero_points, %qmin, %qmax, %dtype, %group_size
      : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>,
        !torch.vtensor<[?,?],si64>, !torch.int, !torch.int, !torch.int,
        !torch.int -> !torch.vtensor<[?,?],si8>
  %out = torch.quantized_decomposed.dequantize_per_channel_group
      %quantized, %scales, %zero_points, %qmin, %qmax, %dtype, %group_size, %out_dtype
      : !torch.vtensor<[?,?],si8>, !torch.vtensor<[?,?],f32>,
        !torch.vtensor<[?,?],si64>, !torch.int, !torch.int, !torch.int,
        !torch.int, !torch.int -> !torch.vtensor<[?,?],f32>
  return %out : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL: func.func @dequantize_per_channel_make_quantized_unsigned(
// CHECK: ^bb0(%[[IN:.*]]: i8, %{{.*}}: f32, %[[ZP:.*]]: i8, %{{.*}}: f32):
// CHECK:   arith.extui %[[ZP]] : i8 to i16
// CHECK:   arith.extui %[[IN]] : i8 to i16
func.func @dequantize_per_channel_make_quantized_unsigned(
    %input: !torch.vtensor<[1,3,3,2],ui8>,
    %scales: !torch.vtensor<[3],f32>,
    %zero_points: !torch.vtensor<[3],ui8>)
    -> !torch.vtensor<[1,3,3,2],f32> {
  %axis = torch.constant.int 1
  %q = torch.aten._make_per_channel_quantized_tensor %input, %scales, %zero_points, %axis
      : !torch.vtensor<[1,3,3,2],ui8>, !torch.vtensor<[3],f32>,
        !torch.vtensor<[3],ui8>, !torch.int -> !torch.vtensor<[1,3,3,2],!torch.quint8>
  %out = torch.aten.dequantize.self %q
      : !torch.vtensor<[1,3,3,2],!torch.quint8> -> !torch.vtensor<[1,3,3,2],f32>
  return %out : !torch.vtensor<[1,3,3,2],f32>
}

// -----

// CHECK-LABEL: func.func @dequantize_per_channel_group_unsigned_zero_point(
// CHECK: ^bb0(%[[IN:.*]]: i8, %{{.*}}: f32, %[[ZP:.*]]: i8, %{{.*}}: f32):
// CHECK-DAG: arith.extui %[[ZP]] : i8 to i16
// CHECK-DAG: arith.extui %[[IN]] : i8 to i16
func.func @dequantize_per_channel_group_unsigned_zero_point(
    %input: !torch.vtensor<[4,16],ui8>,
    %scales: !torch.vtensor<[4,4],f32>,
    %zero_points: !torch.vtensor<[4,4],ui8>)
    -> !torch.vtensor<[4,16],f32> {
  %qmin = torch.constant.int 0
  %qmax = torch.constant.int 255
  %dtype = torch.constant.int 0
  %group_size = torch.constant.int 4
  %out_dtype = torch.constant.int 6
  %out = torch.quantized_decomposed.dequantize_per_channel_group
      %input, %scales, %zero_points, %qmin, %qmax, %dtype, %group_size, %out_dtype
      : !torch.vtensor<[4,16],ui8>, !torch.vtensor<[4,4],f32>,
        !torch.vtensor<[4,4],ui8>, !torch.int, !torch.int, !torch.int,
        !torch.int, !torch.int -> !torch.vtensor<[4,16],f32>
  return %out : !torch.vtensor<[4,16],f32>
}

// -----

// CHECK-LABEL: func.func @quantize_per_channel_group_unsigned_zero_point(
// CHECK: ^bb0(%{{.*}}: f32, %{{.*}}: f32, %[[ZP:.*]]: i8, %{{.*}}: i8):
// CHECK:   arith.uitofp %[[ZP]] : i8 to f32
func.func @quantize_per_channel_group_unsigned_zero_point(
    %input: !torch.vtensor<[4,16],f32>,
    %scales: !torch.vtensor<[4,4],f32>,
    %zero_points: !torch.vtensor<[4,4],ui8>)
    -> !torch.vtensor<[4,16],ui8> {
  %qmin = torch.constant.int 0
  %qmax = torch.constant.int 255
  %dtype = torch.constant.int 0
  %group_size = torch.constant.int 4
  %out = torch.quantized_decomposed.quantize_per_channel_group
      %input, %scales, %zero_points, %qmin, %qmax, %dtype, %group_size
      : !torch.vtensor<[4,16],f32>, !torch.vtensor<[4,4],f32>,
        !torch.vtensor<[4,4],ui8>, !torch.int, !torch.int, !torch.int,
        !torch.int -> !torch.vtensor<[4,16],ui8>
  return %out : !torch.vtensor<[4,16],ui8>
}


// -----

// Per-token quantization: scales/ZP shape matches input with last dim = 1.
// CHECK: #[[IDENTITY:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[TOKEN:.*]] = affine_map<(d0, d1) -> (d0, 0)>
// CHECK-LABEL: func.func @quantize_per_token(
// CHECK: %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[IDENTITY]], #[[TOKEN]], #[[TOKEN]], #[[IDENTITY]]]
func.func @quantize_per_token(
    %input: !torch.vtensor<[4,16],f32>,
    %scales: !torch.vtensor<[4,1],f32>,
    %zero_points: !torch.vtensor<[4,1],si64>)
    -> !torch.vtensor<[4,16],si8> {
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 2
  %out = torch.quantized_decomposed.quantize_per_token
      %input, %scales, %zero_points, %qmin, %qmax, %dtype
      : !torch.vtensor<[4,16],f32>, !torch.vtensor<[4,1],f32>,
        !torch.vtensor<[4,1],si64>, !torch.int, !torch.int, !torch.int
      -> !torch.vtensor<[4,16],si8>
  return %out : !torch.vtensor<[4,16],si8>
}

// -----

// Per-token dequantization.
// CHECK: #[[IDENTITY:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[TOKEN:.*]] = affine_map<(d0, d1) -> (d0, 0)>
// CHECK-LABEL: func.func @dequantize_per_token(
// CHECK: %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[IDENTITY]], #[[TOKEN]], #[[TOKEN]], #[[IDENTITY]]]
func.func @dequantize_per_token(
    %input: !torch.vtensor<[4,16],si8>,
    %scales: !torch.vtensor<[4,1],f32>,
    %zero_points: !torch.vtensor<[4,1],si64>)
    -> !torch.vtensor<[4,16],f32> {
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 2
  %out_dtype = torch.constant.int 6
  %out = torch.quantized_decomposed.dequantize_per_token
      %input, %scales, %zero_points, %qmin, %qmax, %dtype, %out_dtype
      : !torch.vtensor<[4,16],si8>, !torch.vtensor<[4,1],f32>,
        !torch.vtensor<[4,1],si64>, !torch.int, !torch.int, !torch.int,
        !torch.int -> !torch.vtensor<[4,16],f32>
  return %out : !torch.vtensor<[4,16],f32>
}

// -----

// 3D per-token quantization.
// CHECK: #[[IDENTITY:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[TOKEN:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
// CHECK-LABEL: func.func @quantize_per_token_3d(
// CHECK: %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[IDENTITY]], #[[TOKEN]], #[[TOKEN]], #[[IDENTITY]]]
func.func @quantize_per_token_3d(
    %input: !torch.vtensor<[2,4,16],f32>,
    %scales: !torch.vtensor<[2,4,1],f32>,
    %zero_points: !torch.vtensor<[2,4,1],si64>)
    -> !torch.vtensor<[2,4,16],si8> {
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 2
  %out = torch.quantized_decomposed.quantize_per_token
      %input, %scales, %zero_points, %qmin, %qmax, %dtype
      : !torch.vtensor<[2,4,16],f32>, !torch.vtensor<[2,4,1],f32>,
        !torch.vtensor<[2,4,1],si64>, !torch.int, !torch.int, !torch.int
      -> !torch.vtensor<[2,4,16],si8>
  return %out : !torch.vtensor<[2,4,16],si8>
}

// -----

// choose_qparams_per_token_asymmetric
// CHECK: #[[IDENTITY:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[TOKEN:.*]] = affine_map<(d0, d1) -> (d0, 0)>
// CHECK-LABEL: func.func @choose_qparams_per_token_asymmetric(
// CHECK-SAME:    %[[ARG0:.*]]: !torch.vtensor<[4,16],f32>
//
// init fills with +inf for min-reduction, -inf for max-reduction
// CHECK-DAG:   %[[INF:.*]] = arith.constant 0x7F800000 : f32
// CHECK-DAG:   %[[NEGINF:.*]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:   %[[FILL_MIN:.*]] = linalg.fill ins(%[[INF]] : f32)
// CHECK-DAG:   %[[FILL_MAX:.*]] = linalg.fill ins(%[[NEGINF]] : f32)
//
// first generic: reduce each token to per-token-min and per-token-max
// CHECK:       %[[RED:.*]]:2 = linalg.generic
// CHECK-SAME:    indexing_maps = [#[[IDENTITY]], #[[TOKEN]], #[[TOKEN]]]
// CHECK-SAME:    ins(%{{.*}} : tensor<4x16xf32>)
// CHECK:       ^bb0(%[[IN:.*]]: f32, %[[OUT_MIN:.*]]: f32, %[[OUT_MAX:.*]]: f32):
// CHECK:         %[[TMIN:.*]] = arith.minimumf %[[IN]], %[[OUT_MIN]] : f32
// CHECK:         %[[TMAX:.*]] = arith.maximumf %[[IN]], %[[OUT_MAX]] : f32
// CHECK:         linalg.yield %[[TMIN]], %[[TMAX]] : f32, f32
//
// Second generic: compute scale and zero-point
// CHECK:       %[[QP:.*]]:2 = linalg.generic
// CHECK-SAME:    indexing_maps = [#[[IDENTITY]], #[[IDENTITY]], #[[IDENTITY]], #[[IDENTITY]]]
// CHECK-SAME:    ins(%[[RED]]#0, %[[RED]]#1 : tensor<4x1xf32>, tensor<4x1xf32>)
// CHECK:       ^bb0(%[[PMIN:.*]]: f32, %[[PMAX:.*]]: f32, %{{.*}}: f32, %{{.*}}: i32):
// CHECK:         %[[C255:.*]]   = arith.constant 2.550000e+02 : f32
// CHECK:         %[[EPS:.*]]    = arith.constant 1.1920929E-7 : f32
// CHECK:         %[[QMIN_F:.*]] = arith.constant -1.280000e+02 : f32
// CHECK:         %[[QMAX_F:.*]] = arith.constant 1.270000e+02 : f32
// CHECK:         %[[ZERO:.*]]   = arith.constant 0.000000e+00 : f32
// clamp min to <= 0, max to >= 0
// CHECK:         %[[CMIN:.*]] = arith.minimumf %[[PMIN]], %[[ZERO]] : f32
// CHECK:         %[[CMAX:.*]] = arith.maximumf %[[PMAX]], %[[ZERO]] : f32
// range = max - min
// CHECK:         %[[RANGE:.*]] = arith.subf %[[CMAX]], %[[CMIN]] : f32
// CHECK:         %[[S0:.*]]    = arith.divf %[[RANGE]], %[[C255]] : f32
// clamp scale away from zero
// CHECK:         %[[SCALE:.*]] = arith.maximumf %[[S0]], %[[EPS]] : f32
// zero-point candidates from both sides: zp_min = min/scale, zp_max = max/scale
// CHECK:         %[[ZP_MIN:.*]] = arith.divf %[[CMIN]], %[[SCALE]] : f32
// CHECK:         %[[ZP_MAX:.*]] = arith.divf %[[CMAX]], %[[SCALE]] : f32
// CHECK:         %[[A:.*]]   = arith.addf %[[QMIN_F]], %[[ZP_MIN]] : f32
// CHECK:         %[[B:.*]]   = arith.addf %[[QMAX_F]], %[[ZP_MAX]] : f32
// CHECK:         %[[SUM:.*]] = arith.addf %[[A]], %[[B]] : f32
// pick the candidate direction based on sign of sum
// CHECK:         %[[PRED:.*]]  = arith.cmpf ogt, %[[SUM]], %[[ZERO]] : f32
// CHECK:         %[[CND0:.*]]  = arith.subf %[[QMIN_F]], %[[ZP_MIN]] : f32
// CHECK:         %[[CND1:.*]]  = arith.subf %[[QMAX_F]], %[[ZP_MAX]] : f32
// CHECK:         %[[SEL:.*]]   = arith.select %[[PRED]], %[[CND0]], %[[CND1]] : f32
// clamp zero-point to [qmin, qmax]
// CHECK:         %[[CLO:.*]]   = arith.maximumf %[[SEL]], %[[QMIN_F]] : f32
// CHECK:         %[[CHI:.*]]   = arith.minimumf %[[CLO]], %[[QMAX_F]] : f32
// CHECK:         %[[RND:.*]]   = math.roundeven %[[CHI]] : f32
// CHECK:         %[[ZP_I32:.*]] = arith.fptosi %[[RND]] : f32 to i32
// scale is yielded first, zero-point second
// CHECK:         linalg.yield %[[SCALE]], %[[ZP_I32]] : f32, i32
func.func @choose_qparams_per_token_asymmetric(
    %input: !torch.vtensor<[4,16],f32>)
    -> (!torch.vtensor<[4,1],f32>, !torch.vtensor<[4,1],si32>) {
  %dtype = torch.constant.int 1
  %scale, %zp = torch.quantized_decomposed.choose_qparams_per_token_asymmetric
      %input, %dtype
      : !torch.vtensor<[4,16],f32>, !torch.int
      -> !torch.vtensor<[4,1],f32>, !torch.vtensor<[4,1],si32>
  return %scale, %zp : !torch.vtensor<[4,1],f32>, !torch.vtensor<[4,1],si32>
}

// -----

// choose_qparams_per_token
// CHECK: #[[IDENTITY:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[TOKEN:.*]] = affine_map<(d0, d1) -> (d0, 0)>
// CHECK-LABEL: func.func @choose_qparams_per_token(
// CHECK-SAME:    %[[ARG0:.*]]: !torch.vtensor<[4,16],f32>
//
// CHECK:       %[[ZERO_CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:       %[[AMAX_INIT:.*]] = linalg.fill ins(%[[ZERO_CST]] : f32)
//
// First generic: compute per-token absolute maximum.
// CHECK:       %[[AMAX:.*]] = linalg.generic
// CHECK-SAME:    indexing_maps = [#[[IDENTITY]], #[[TOKEN]]]
// CHECK-SAME:    ins(%{{.*}} : tensor<4x16xf32>)
// CHECK:       ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: f32):
// abs(input) then running max with accumulator.
// CHECK:         %[[ABS:.*]] = math.absf %[[IN]] : f32
// CHECK:         %[[MAX:.*]] = arith.maximumf %[[ABS]], %[[OUT]] : f32
// CHECK:         linalg.yield %[[MAX]] : f32
//
// CHECK:       %[[CAST:.*]] = tensor.cast %[[AMAX]] : tensor<?x1xf32> to tensor<4x1xf32>
// Second generic: scale = max(amax, eps) / qmax; zero-point is always 0.
// CHECK:       %[[QP:.*]]:2 = linalg.generic
// CHECK-SAME:    indexing_maps = [#[[IDENTITY]], #[[IDENTITY]], #[[IDENTITY]]]
// CHECK-SAME:    ins(%[[CAST]] : tensor<4x1xf32>)
// CHECK:       ^bb0(%[[IN2:.*]]: f32, %{{.*}}: f32, %{{.*}}: i32):
// CHECK:         %[[QMAX_F:.*]] = arith.constant 1.270000e+02 : f32
// CHECK:         %[[EPS:.*]]    = arith.constant 9.99999974E-6 : f32
// CHECK:         %[[ZP_C:.*]]   = arith.constant 0 : i32
// CHECK:         %[[CLAMPED:.*]] = arith.maximumf %[[IN2]], %[[EPS]] : f32
// scale = clamped_amax / qmax
// CHECK:         %[[SCALE:.*]]  = arith.divf %[[CLAMPED]], %[[QMAX_F]] : f32
// CHECK:         linalg.yield %[[SCALE]], %[[ZP_C]] : f32, i32
func.func @choose_qparams_per_token(
  %input: !torch.vtensor<[4,16],f32>)
    -> (!torch.vtensor<[4,1],f32>, !torch.vtensor<[4,1],si32>) {
  %dtype = torch.constant.int 1
  %scale, %zp = torch.quantized_decomposed.choose_qparams_per_token
      %input, %dtype
      : !torch.vtensor<[4,16],f32>, !torch.int
      -> !torch.vtensor<[4,1],f32>, !torch.vtensor<[4,1],si32>
  return %scale, %zp : !torch.vtensor<[4,1],f32>, !torch.vtensor<[4,1],si32>
}

// -----

// Dynamic per token symmetric quant flow: verifies affine maps and tensor shapes for
// dynamic dimensions.
// CHECK: #[[IDENTITY2:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[TOKEN2:.*]] = affine_map<(d0, d1) -> (d0, 0)>
// CHECK-LABEL: func.func @dynamic_per_token_symmetric_choose_quant_roundtrip(
// CHECK-SAME:    %[[ARG0:.*]]: !torch.vtensor<[?,?],f32>
// abs-max reduction: identity input mapped to token output
// CHECK:       linalg.fill
// CHECK:       linalg.generic
// CHECK-SAME:    indexing_maps = [#[[IDENTITY2]], #[[TOKEN2]]]
// CHECK-SAME:    ins(%{{.*}} : tensor<?x?xf32>) outs(%{{.*}} : tensor<?x1xf32>)
// scale derivation: token-shaped input, token-shaped outputs
// CHECK:       linalg.generic
// CHECK-SAME:    indexing_maps = [#[[IDENTITY2]], #[[IDENTITY2]], #[[IDENTITY2]]]
// CHECK-SAME:    ins(%{{.*}} : tensor<?x1xf32>) outs(%{{.*}} : tensor<?x1xf32>, tensor<?x1xi32>)
// quantize: identity input, token scale+zp, identity output
// CHECK:       linalg.generic
// CHECK-SAME:    indexing_maps = [#[[IDENTITY2]], #[[TOKEN2]], #[[TOKEN2]], #[[IDENTITY2]]]
// CHECK-SAME:    ins(%{{.*}} : tensor<?x?xf32>, tensor<?x1xf32>, tensor<?x1xi32>) outs(%{{.*}} : tensor<?x?xi8>)
// dequantize: identity input, token scale+zp, identity output
// CHECK:       linalg.generic
// CHECK-SAME:    indexing_maps = [#[[IDENTITY2]], #[[TOKEN2]], #[[TOKEN2]], #[[IDENTITY2]]]
// CHECK-SAME:    ins(%{{.*}} : tensor<?x?xi8>, tensor<?x1xf32>, tensor<?x1xi32>) outs(%{{.*}} : tensor<?x?xf32>)
func.func @dynamic_per_token_symmetric_choose_quant_roundtrip(
    %input: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %dtype = torch.constant.int 1
  %qmin  = torch.constant.int -128
  %qmax  = torch.constant.int 127
  %qdtype = torch.constant.int 2
  %out_dtype = torch.constant.int 6
  %scale, %zp = torch.quantized_decomposed.choose_qparams_per_token
      %input, %dtype
      : !torch.vtensor<[?,?],f32>, !torch.int
      -> !torch.vtensor<[?,1],f32>, !torch.vtensor<[?,1],si32>
  %quantized = torch.quantized_decomposed.quantize_per_token
      %input, %scale, %zp, %qmin, %qmax, %qdtype
      : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,1],f32>,
        !torch.vtensor<[?,1],si32>, !torch.int, !torch.int, !torch.int
      -> !torch.vtensor<[?,?],si8>
  %out = torch.quantized_decomposed.dequantize_per_token
      %quantized, %scale, %zp, %qmin, %qmax, %qdtype, %out_dtype
      : !torch.vtensor<[?,?],si8>, !torch.vtensor<[?,1],f32>,
        !torch.vtensor<[?,1],si32>, !torch.int, !torch.int, !torch.int,
        !torch.int -> !torch.vtensor<[?,?],f32>
  return %out : !torch.vtensor<[?,?],f32>
}

// -----

// Dynamic per token asymmetric quant flow: verifies affine maps and tensor shapes for
// dynamic dimensions.
// CHECK: #[[IDENTITY3:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[TOKEN3:.*]] = affine_map<(d0, d1) -> (d0, 0)>
// CHECK-LABEL: func.func @dynamic_per_token_asymmetric_choose_quant_roundtrip(
// CHECK-SAME:    %[[ARG0:.*]]: !torch.vtensor<[?,?],f32>
// min/max reduction fills: +inf init for min, -inf init for max
// CHECK-DAG:   linalg.fill
// CHECK-DAG:   linalg.fill
// min/max reduction: identity input, two token outputs
// CHECK:       linalg.generic
// CHECK-SAME:    indexing_maps = [#[[IDENTITY3]], #[[TOKEN3]], #[[TOKEN3]]]
// CHECK-SAME:    ins(%{{.*}} : tensor<?x?xf32>) outs(%{{.*}} : tensor<?x1xf32>, tensor<?x1xf32>)
// scale/zp derivation: two token inputs, two token outputs
// CHECK:       linalg.generic
// CHECK-SAME:    indexing_maps = [#[[IDENTITY3]], #[[IDENTITY3]], #[[IDENTITY3]], #[[IDENTITY3]]]
// CHECK-SAME:    ins(%{{.*}} : tensor<?x1xf32>, tensor<?x1xf32>) outs(%{{.*}} : tensor<?x1xf32>, tensor<?x1xi32>)
// quantize: identity input, token scale+zp, identity output
// CHECK:       linalg.generic
// CHECK-SAME:    indexing_maps = [#[[IDENTITY3]], #[[TOKEN3]], #[[TOKEN3]], #[[IDENTITY3]]]
// CHECK-SAME:    ins(%{{.*}} : tensor<?x?xf32>, tensor<?x1xf32>, tensor<?x1xi32>) outs(%{{.*}} : tensor<?x?xi8>)
// dequantize: identity input, token scale+zp, identity output
// CHECK:       linalg.generic
// CHECK-SAME:    indexing_maps = [#[[IDENTITY3]], #[[TOKEN3]], #[[TOKEN3]], #[[IDENTITY3]]]
// CHECK-SAME:    ins(%{{.*}} : tensor<?x?xi8>, tensor<?x1xf32>, tensor<?x1xi32>) outs(%{{.*}} : tensor<?x?xf32>)
func.func @dynamic_per_token_asymmetric_choose_quant_roundtrip(
    %input: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %dtype = torch.constant.int 1
  %qmin  = torch.constant.int -128
  %qmax  = torch.constant.int 127
  %qdtype = torch.constant.int 2
  %out_dtype = torch.constant.int 6
  %scale, %zp = torch.quantized_decomposed.choose_qparams_per_token_asymmetric
      %input, %dtype
      : !torch.vtensor<[?,?],f32>, !torch.int
      -> !torch.vtensor<[?,1],f32>, !torch.vtensor<[?,1],si32>
  %quantized = torch.quantized_decomposed.quantize_per_token
      %input, %scale, %zp, %qmin, %qmax, %qdtype
      : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,1],f32>,
        !torch.vtensor<[?,1],si32>, !torch.int, !torch.int, !torch.int
      -> !torch.vtensor<[?,?],si8>
  %out = torch.quantized_decomposed.dequantize_per_token
      %quantized, %scale, %zp, %qmin, %qmax, %qdtype, %out_dtype
      : !torch.vtensor<[?,?],si8>, !torch.vtensor<[?,1],f32>,
        !torch.vtensor<[?,1],si32>, !torch.int, !torch.int, !torch.int,
        !torch.int -> !torch.vtensor<[?,?],f32>
  return %out : !torch.vtensor<[?,?],f32>
}
