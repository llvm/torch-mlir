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
// CHECK:         %[[SUBF:.*]] = arith.sitofp %[[SUB]] : i64 to f32
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
  %dtype = torch.constant.int 1
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
  %dtype = torch.constant.int 1
  %out = torch.quantized_decomposed.quantize_per_tensor
      %input, %scale, %zp, %qmin, %qmax, %dtype
      : !torch.vtensor<[4,8],f32>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int
      -> !torch.vtensor<[4,8],si8>
  return %out : !torch.vtensor<[4,8],si8>
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
  %dtype = torch.constant.int 1
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
  %dtype = torch.constant.int 1
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
  %dtype = torch.constant.int 1
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
  %dtype = torch.constant.int 1
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
  %dtype = torch.constant.int 1
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
  %dtype = torch.constant.int 1
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

// f16 scales are extended to f32, in which the multiply runs.
// CHECK-LABEL: func.func @dequantize_per_channel_scale_extension(
// CHECK: ^bb0(%[[IN:.*]]: i8, %[[SCALE:.*]]: f16, %{{.*}}: f32):
// CHECK:   %[[FP:.*]] = arith.sitofp %[[IN]] : i8 to f32
// CHECK:   %[[EXT_SCALE:.*]] = arith.extf %[[SCALE]] : f16 to f32
// CHECK:   %[[MUL:.*]] = arith.mulf %[[FP]], %[[EXT_SCALE]] : f32
// CHECK:   linalg.yield %[[MUL]] : f32
func.func @dequantize_per_channel_scale_extension(
    %input: !torch.vtensor<[4,8],si8>,
    %scales: !torch.vtensor<[4],f16>)
    -> !torch.vtensor<[4,8],f32> {
  %axis = torch.constant.int 0
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 1
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
  %dtype = torch.constant.int 1
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
  %dtype = torch.constant.int 1
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
  %dtype = torch.constant.int 1
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
  %dtype = torch.constant.int 1
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
  %dtype = torch.constant.int 1
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

// Per-channel-group with f16 scales and f32 output: the multiply runs in f32.
// CHECK-LABEL: func.func @dequantize_per_channel_group_scale_extension(
// CHECK: ^bb0(%[[IN:.*]]: i8, %[[SCALE:.*]]: f16, %{{.*}}: f32):
// CHECK:   %[[FP:.*]] = arith.sitofp %[[IN]] : i8 to f32
// CHECK:   %[[EXT_SCALE:.*]] = arith.extf %[[SCALE]] : f16 to f32
// CHECK:   %[[MUL:.*]] = arith.mulf %[[FP]], %[[EXT_SCALE]] : f32
// CHECK:   linalg.yield %[[MUL]] : f32
func.func @dequantize_per_channel_group_scale_extension(
    %input: !torch.vtensor<[4,16],si8>,
    %scales: !torch.vtensor<[4,4],f16>)
    -> !torch.vtensor<[4,16],f32> {
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 1
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
  %dtype = torch.constant.int 1
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

// Per-channel-group with a non-integral floating-point zero point, as HQQ
// quantizers produce: subtract in f32, keeping the fraction, then truncate to
// the f16 output.
// onnxruntime's HQQ quantizer optimizes the zero point to a mean:
// https://github.com/microsoft/onnxruntime/blob/69eeb2bb4aba4bf97f091d529222f1516dc98e49/onnxruntime/python/tools/quantization/matmul_nbits_quantizer.py#L581
// CHECK-LABEL: func.func @dequantize_per_channel_group_float_zero_point(
// CHECK: ^bb0(%[[IN:.*]]: i8, %[[SCALE:.*]]: f16, %[[ZP:.*]]: f16, %{{.*}}: f16):
// CHECK:   %[[FP:.*]] = arith.uitofp %[[IN]] : i8 to f32
// CHECK:   %[[EXT_ZP:.*]] = arith.extf %[[ZP]] : f16 to f32
// CHECK:   %[[SUB:.*]] = arith.subf %[[FP]], %[[EXT_ZP]] : f32
// CHECK:   %[[EXT_SCALE:.*]] = arith.extf %[[SCALE]] : f16 to f32
// CHECK:   %[[MUL:.*]] = arith.mulf %[[SUB]], %[[EXT_SCALE]] : f32
// CHECK:   %[[TRUNC:.*]] = arith.truncf %[[MUL]] : f32 to f16
// CHECK:   linalg.yield %[[TRUNC]] : f16
func.func @dequantize_per_channel_group_float_zero_point(
    %input: !torch.vtensor<[4,16],ui8>,
    %scales: !torch.vtensor<[4,4],f16>,
    %zero_points: !torch.vtensor<[4,4],f16>)
    -> !torch.vtensor<[4,16],f16> {
  %qmin = torch.constant.int 0
  %qmax = torch.constant.int 15
  %dtype = torch.constant.int 0
  %group_size = torch.constant.int 4
  %out_dtype = torch.constant.int 5
  %out = torch.quantized_decomposed.dequantize_per_channel_group
      %input, %scales, %zero_points, %qmin, %qmax, %dtype, %group_size, %out_dtype
      : !torch.vtensor<[4,16],ui8>, !torch.vtensor<[4,4],f16>,
        !torch.vtensor<[4,4],f16>, !torch.int, !torch.int, !torch.int,
        !torch.int, !torch.int -> !torch.vtensor<[4,16],f16>
  return %out : !torch.vtensor<[4,16],f16>
}

// -----

// choose_qparams_per_token's meta function types the scales as f64, which
// FX-imported graphs carry; the scales are truncated so the dequantize stays in
// f32:
// https://github.com/pytorch/pytorch/blob/f3c0cf56a7577d677c37718c4edc41b8fc7fb51a/torch/ao/quantization/fx/_decomposed.py#L818-L825
// CHECK-LABEL: func.func @dequantize_per_token_f64_scales(
// CHECK: ^bb0(%[[IN:.*]]: i8, %[[SCALE:.*]]: f64, %[[ZP:.*]]: i64, %{{.*}}: f32):
// CHECK:   %[[FP:.*]] = arith.sitofp %{{.*}} : i64 to f32
// CHECK:   %[[TRUNC_SCALE:.*]] = arith.truncf %[[SCALE]] : f64 to f32
// CHECK:   %[[MUL:.*]] = arith.mulf %[[FP]], %[[TRUNC_SCALE]] : f32
// CHECK:   linalg.yield %[[MUL]] : f32
func.func @dequantize_per_token_f64_scales(
    %input: !torch.vtensor<[4,16],si8>,
    %scales: !torch.vtensor<[4,1],f64>,
    %zero_points: !torch.vtensor<[4,1],si64>)
    -> !torch.vtensor<[4,16],f32> {
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 1
  %out_dtype = torch.constant.int 6
  %out = torch.quantized_decomposed.dequantize_per_token
      %input, %scales, %zero_points, %qmin, %qmax, %dtype, %out_dtype
      : !torch.vtensor<[4,16],si8>, !torch.vtensor<[4,1],f64>,
        !torch.vtensor<[4,1],si64>, !torch.int, !torch.int, !torch.int,
        !torch.int -> !torch.vtensor<[4,16],f32>
  return %out : !torch.vtensor<[4,16],f32>
}

// -----

// The quantize side of the f64 scales above also runs in f32: the f16 input is
// extended instead of the scales being truncated to f16.
// CHECK-LABEL: func.func @quantize_per_token_f64_scales(
// CHECK: ^bb0(%[[IN:.*]]: f16, %[[SCALE:.*]]: f64, %[[ZP:.*]]: i64, %{{.*}}: i8):
// CHECK:   %[[EXT_IN:.*]] = arith.extf %[[IN]] : f16 to f32
// CHECK:   %[[TRUNC_SCALE:.*]] = arith.truncf %[[SCALE]] : f64 to f32
// CHECK:   %[[DIV:.*]] = arith.divf %[[EXT_IN]], %[[TRUNC_SCALE]] : f32
// CHECK:   arith.fptosi %{{.*}} : f32 to i8
func.func @quantize_per_token_f64_scales(
    %input: !torch.vtensor<[4,16],f16>,
    %scales: !torch.vtensor<[4,1],f64>,
    %zero_points: !torch.vtensor<[4,1],si64>)
    -> !torch.vtensor<[4,16],si8> {
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 1
  %out = torch.quantized_decomposed.quantize_per_token
      %input, %scales, %zero_points, %qmin, %qmax, %dtype
      : !torch.vtensor<[4,16],f16>, !torch.vtensor<[4,1],f64>,
        !torch.vtensor<[4,1],si64>, !torch.int, !torch.int, !torch.int
      -> !torch.vtensor<[4,16],si8>
  return %out : !torch.vtensor<[4,16],si8>
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
  %dtype = torch.constant.int 1
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
  %dtype = torch.constant.int 1
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
  %dtype = torch.constant.int 1
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
  %dtype = torch.constant.int 1
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

// Per-channel quantization with floating-point zero points. Quantize adds the
// zero point after rounding; dequantize subtracts it in f32.
// CHECK-LABEL: func.func @per_channel_float_zero_points(
// CHECK: ^bb0(%[[IN:.*]]: f32, %[[SCALE:.*]]: f32, %[[ZP:.*]]: f32, %{{.*}}: i8):
// CHECK:   %[[DIV:.*]] = arith.divf %[[IN]], %[[SCALE]] : f32
// CHECK:   %[[RND:.*]] = math.roundeven %[[DIV]] : f32
// CHECK:   arith.addf %[[RND]], %[[ZP]] : f32
// CHECK: ^bb0(%[[Q:.*]]: i8, %[[DQ_SCALE:.*]]: f32, %[[DQ_ZP:.*]]: f32, %{{.*}}: f32):
// CHECK:   %[[FP:.*]] = arith.uitofp %[[Q]] : i8 to f32
// CHECK:   %[[SUB:.*]] = arith.subf %[[FP]], %[[DQ_ZP]] : f32
// CHECK:   arith.mulf %[[SUB]], %[[DQ_SCALE]] : f32
func.func @per_channel_float_zero_points(
    %input: !torch.vtensor<[4,8],f32>,
    %scales: !torch.vtensor<[4],f32>,
    %zero_points: !torch.vtensor<[4],f32>)
    -> !torch.vtensor<[4,8],f32> {
  %axis = torch.constant.int 0
  %qmin = torch.constant.int 0
  %qmax = torch.constant.int 255
  %dtype = torch.constant.int 0
  %none = torch.constant.none
  %q = torch.quantized_decomposed.quantize_per_channel
      %input, %scales, %zero_points, %axis, %qmin, %qmax, %dtype
      : !torch.vtensor<[4,8],f32>, !torch.vtensor<[4],f32>,
        !torch.vtensor<[4],f32>, !torch.int, !torch.int, !torch.int,
        !torch.int -> !torch.vtensor<[4,8],ui8>
  %dq = torch.quantized_decomposed.dequantize_per_channel
      %q, %scales, %zero_points, %axis, %qmin, %qmax, %dtype, %none
      : !torch.vtensor<[4,8],ui8>, !torch.vtensor<[4],f32>,
        !torch.vtensor<[4],f32>, !torch.int, !torch.int, !torch.int,
        !torch.int, !torch.none -> !torch.vtensor<[4,8],f32>
  return %dq : !torch.vtensor<[4,8],f32>
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

// Per-token dequantization with f32 zero points, the dtype eager
// choose_qparams_per_token returns (its meta function reports si64):
// https://github.com/pytorch/pytorch/blob/f3c0cf56a7577d677c37718c4edc41b8fc7fb51a/torch/ao/quantization/fx/_decomposed.py#L809
// https://github.com/pytorch/pytorch/blob/f3c0cf56a7577d677c37718c4edc41b8fc7fb51a/torch/ao/quantization/fx/_decomposed.py#L818-L825
// CHECK-LABEL: func.func @dequantize_per_token_float_zero_point(
// CHECK: ^bb0(%[[IN:.*]]: i8, %[[SCALE:.*]]: f32, %[[ZP:.*]]: f32, %{{.*}}: f32):
// CHECK:   %[[FP:.*]] = arith.sitofp %[[IN]] : i8 to f32
// CHECK:   %[[SUB:.*]] = arith.subf %[[FP]], %[[ZP]] : f32
// CHECK:   %[[MUL:.*]] = arith.mulf %[[SUB]], %[[SCALE]] : f32
// CHECK:   linalg.yield %[[MUL]] : f32
func.func @dequantize_per_token_float_zero_point(
    %input: !torch.vtensor<[4,16],si8>,
    %scales: !torch.vtensor<[4,1],f32>,
    %zero_points: !torch.vtensor<[4,1],f32>)
    -> !torch.vtensor<[4,16],f32> {
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 1
  %out_dtype = torch.constant.int 6
  %out = torch.quantized_decomposed.dequantize_per_token
      %input, %scales, %zero_points, %qmin, %qmax, %dtype, %out_dtype
      : !torch.vtensor<[4,16],si8>, !torch.vtensor<[4,1],f32>,
        !torch.vtensor<[4,1],f32>, !torch.int, !torch.int, !torch.int,
        !torch.int -> !torch.vtensor<[4,16],f32>
  return %out : !torch.vtensor<[4,16],f32>
}

// -----

// Per-token quantization of an f16 input with f32 scales computes in f32
// instead of truncating the scale to f16.
// CHECK-LABEL: func.func @quantize_per_token_f16_input(
// CHECK: ^bb0(%[[IN:.*]]: f16, %[[SCALE:.*]]: f32, %[[ZP:.*]]: i64, %{{.*}}: i8):
// CHECK:   %[[EXT_IN:.*]] = arith.extf %[[IN]] : f16 to f32
// CHECK:   %[[DIV:.*]] = arith.divf %[[EXT_IN]], %[[SCALE]] : f32
// CHECK:   arith.fptosi %{{.*}} : f32 to i8
func.func @quantize_per_token_f16_input(
    %input: !torch.vtensor<[4,16],f16>,
    %scales: !torch.vtensor<[4,1],f32>,
    %zero_points: !torch.vtensor<[4,1],si64>)
    -> !torch.vtensor<[4,16],si8> {
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 1
  %out = torch.quantized_decomposed.quantize_per_token
      %input, %scales, %zero_points, %qmin, %qmax, %dtype
      : !torch.vtensor<[4,16],f16>, !torch.vtensor<[4,1],f32>,
        !torch.vtensor<[4,1],si64>, !torch.int, !torch.int, !torch.int
      -> !torch.vtensor<[4,16],si8>
  return %out : !torch.vtensor<[4,16],si8>
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

// -----

// CHECK: #[[IDENTITY:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[TOKEN:.*]] = affine_map<(d0, d1, d2) -> (0, d1)>
// CHECK-LABEL: func.func @quantize_affine_per_token(
// CHECK-SAME: %[[INPUT:.*]]: !torch.vtensor<[1,4,8],bf16>, %[[SCALES:.*]]: !torch.vtensor<[1,4],f32>, %[[ZPS:.*]]: !torch.vtensor<[1,4],f32>)
// CHECK-DAG: %[[INPUT_T:.*]] = torch_c.to_builtin_tensor %[[INPUT]]
// CHECK-DAG: %[[SCALES_T:.*]] = torch_c.to_builtin_tensor %[[SCALES]]
// CHECK-DAG: %[[ZPS_T:.*]] = torch_c.to_builtin_tensor %[[ZPS]]
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<1x4x8xi8>
// CHECK: %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[IDENTITY]], #[[TOKEN]], #[[TOKEN]], #[[IDENTITY]]]
// CHECK-SAME: ins(%[[INPUT_T]], %[[SCALES_T]], %[[ZPS_T]] : tensor<1x4x8xbf16>, tensor<1x4xf32>, tensor<1x4xf32>)
// CHECK-SAME: outs(%[[EMPTY]] : tensor<1x4x8xi8>)
// CHECK: ^bb0(%[[IN:.*]]: bf16, %[[SCALE:.*]]: f32, %[[ZP:.*]]: f32, %{{.*}}: i8):
// CHECK:   %[[EXT:.*]] = arith.extf %[[IN]] : bf16 to f32
// CHECK:   %[[QMIN:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:   %[[QMAX:.*]] = arith.constant 2.550000e+02 : f32
// CHECK:   %[[DIV:.*]] = arith.divf %[[EXT]], %[[SCALE]] : f32
// CHECK:   %[[ROUND:.*]] = math.roundeven %[[DIV]] : f32
// CHECK:   %[[ADD:.*]] = arith.addf %[[ROUND]], %[[ZP]] : f32
// CHECK:   %[[LOW:.*]] = arith.maximumf %[[ADD]], %[[QMIN]] : f32
// CHECK:   %[[HIGH:.*]] = arith.minimumf %[[LOW]], %[[QMAX]] : f32
// CHECK:   %[[RESULT:.*]] = arith.fptoui %[[HIGH]] : f32 to i8
// CHECK:   linalg.yield %[[RESULT]] : i8
func.func @quantize_affine_per_token(
    %input: !torch.vtensor<[1,4,8],bf16>,
    %scale: !torch.vtensor<[1,4],f32>,
    %zero_point: !torch.vtensor<[1,4],f32>)
    -> !torch.vtensor<[1,4,8],ui8> {
  %int1 = torch.constant.int 1
  %int8 = torch.constant.int 8
  %dtype = torch.constant.int 0
  %qmin = torch.constant.float 0.000000e+00
  %none = torch.constant.none
  %block_size = torch.prim.ListConstruct %int1, %int1, %int8
      : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %out = torch.torchao.quantize_affine
      %input, %block_size, %scale, %zero_point, %dtype, %qmin, %none
      : !torch.vtensor<[1,4,8],bf16>, !torch.list<int>,
        !torch.vtensor<[1,4],f32>, !torch.vtensor<[1,4],f32>, !torch.int,
        !torch.float, !torch.none -> !torch.vtensor<[1,4,8],ui8>
  return %out : !torch.vtensor<[1,4,8],ui8>
}

// -----

// CHECK-LABEL: func.func @quantize_affine_unsigned_zero_point(
// CHECK: ^bb0(%[[IN:.*]]: f32, %[[SCALE:.*]]: bf16, %[[ZP:.*]]: i8, %{{.*}}: i8):
// CHECK:   arith.constant 0.000000e+00 : f32
// CHECK:   arith.constant 1.500000e+01 : f32
// CHECK:   %[[SCALE_EXT:.*]] = arith.extf %[[SCALE]] : bf16 to f32
// CHECK:   arith.uitofp %[[ZP]] : i8 to f32
// CHECK:   arith.divf %[[IN]], %[[SCALE_EXT]] : f32
func.func @quantize_affine_unsigned_zero_point(
    %input: !torch.vtensor<[4,16],f32>,
    %scale: !torch.vtensor<[4,1],bf16>,
    %zero_point: !torch.vtensor<[4,1],ui8>)
    -> !torch.vtensor<[4,16],ui8> {
  %int1 = torch.constant.int 1
  %int16 = torch.constant.int 16
  %dtype = torch.constant.int 0
  %none = torch.constant.none
  %qmax = torch.constant.int 15
  %block_size = torch.prim.ListConstruct %int1, %int16
      : (!torch.int, !torch.int) -> !torch.list<int>
  %out = torch.torchao.quantize_affine
      %input, %block_size, %scale, %zero_point, %dtype, %none, %qmax
      : !torch.vtensor<[4,16],f32>, !torch.list<int>,
        !torch.vtensor<[4,1],bf16>, !torch.vtensor<[4,1],ui8>, !torch.int,
        !torch.none, !torch.int -> !torch.vtensor<[4,16],ui8>
  return %out : !torch.vtensor<[4,16],ui8>
}

// -----

// CHECK-LABEL: func.func @quantize_affine_float_zero_point(
// CHECK: ^bb0(%[[IN:.*]]: bf16, %[[SCALE:.*]]: bf16, %[[ZP:.*]]: f32, %{{.*}}: i16):
// CHECK:   %[[IN_EXT:.*]] = arith.extf %[[IN]] : bf16 to f32
// CHECK:   %[[QMIN:.*]] = arith.constant -3.276800e+04 : f32
// CHECK:   %[[QMAX:.*]] = arith.constant 3.276700e+04 : f32
// CHECK:   %[[SCALE_EXT:.*]] = arith.extf %[[SCALE]] : bf16 to f32
// CHECK:   %[[DIV:.*]] = arith.divf %[[IN_EXT]], %[[SCALE_EXT]] : f32
// CHECK:   %[[ROUND:.*]] = math.roundeven %[[DIV]] : f32
// CHECK:   %[[ADD:.*]] = arith.addf %[[ROUND]], %[[ZP]] : f32
// CHECK:   %[[LOW:.*]] = arith.maximumf %[[ADD]], %[[QMIN]] : f32
// CHECK:   %[[HIGH:.*]] = arith.minimumf %[[LOW]], %[[QMAX]] : f32
// CHECK:   arith.fptosi %[[HIGH]] : f32 to i16
func.func @quantize_affine_float_zero_point(
    %input: !torch.vtensor<[4,16],bf16>,
    %scale: !torch.vtensor<[4,1],bf16>,
    %zero_point: !torch.vtensor<[4,1],f32>)
    -> !torch.vtensor<[4,16],si16> {
  %int1 = torch.constant.int 1
  %int16 = torch.constant.int 16
  %dtype = torch.constant.int 2
  %none = torch.constant.none
  %block_size = torch.prim.ListConstruct %int1, %int16
      : (!torch.int, !torch.int) -> !torch.list<int>
  %out = torch.torchao.quantize_affine
      %input, %block_size, %scale, %zero_point, %dtype, %none, %none
      : !torch.vtensor<[4,16],bf16>, !torch.list<int>,
        !torch.vtensor<[4,1],bf16>, !torch.vtensor<[4,1],f32>, !torch.int,
        !torch.none, !torch.none -> !torch.vtensor<[4,16],si16>
  return %out : !torch.vtensor<[4,16],si16>
}

// -----

// A non-constant block_size entry, such as a symbolic size from torch.export,
// spans its dimension.
// CHECK: #[[IDENTITY:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[TENSOR:.*]] = affine_map<(d0, d1) -> ()>
// CHECK-LABEL: func.func @quantize_affine_per_tensor_dynamic(
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[IDENTITY]], #[[TENSOR]], #[[TENSOR]], #[[IDENTITY]]]
// CHECK-SAME: ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<2x?xf32>, tensor<f32>, tensor<i8>)
func.func @quantize_affine_per_tensor_dynamic(
    %input: !torch.vtensor<[2,?],f32>,
    %scale: !torch.vtensor<[],f32>,
    %zero_point: !torch.vtensor<[],ui8>,
    %size: !torch.int) -> !torch.vtensor<[2,?],ui8> {
  %int2 = torch.constant.int 2
  %dtype = torch.constant.int 0
  %none = torch.constant.none
  %block_size = torch.prim.ListConstruct %int2, %size
      : (!torch.int, !torch.int) -> !torch.list<int>
  %out = torch.torchao.quantize_affine
      %input, %block_size, %scale, %zero_point, %dtype, %none, %none
      : !torch.vtensor<[2,?],f32>, !torch.list<int>, !torch.vtensor<[],f32>,
        !torch.vtensor<[],ui8>, !torch.int, !torch.none, !torch.none
        -> !torch.vtensor<[2,?],ui8>
  return %out : !torch.vtensor<[2,?],ui8>
}

// -----

// CHECK: #[[IDENTITY:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[GROUP:.*]] = affine_map<(d0, d1) -> (d0, d1 floordiv 8)>
// CHECK-LABEL: func.func @dequantize_affine_per_group(
// CHECK-SAME: %[[INPUT:.*]]: !torch.vtensor<[4,16],ui8>
// CHECK-SAME: %[[SCALES:.*]]: !torch.vtensor<[4,2],f32>
// CHECK-SAME: %[[ZPS:.*]]: !torch.vtensor<[4,2],si8>
// CHECK-DAG: %[[INPUT_T:.*]] = torch_c.to_builtin_tensor %[[INPUT]]
// CHECK-DAG: %[[SCALES_T:.*]] = torch_c.to_builtin_tensor %[[SCALES]]
// CHECK-DAG: %[[ZPS_T:.*]] = torch_c.to_builtin_tensor %[[ZPS]]
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<4x16xf32>
// CHECK: %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[IDENTITY]], #[[GROUP]], #[[GROUP]], #[[IDENTITY]]]
// CHECK-SAME: ins(%[[INPUT_T]], %[[SCALES_T]], %[[ZPS_T]] : tensor<4x16xi8>, tensor<4x2xf32>, tensor<4x2xi8>)
// CHECK-SAME: outs(%[[EMPTY]] : tensor<4x16xf32>)
// CHECK: ^bb0(%[[IN:.*]]: i8, %[[SCALE:.*]]: f32, %[[ZP:.*]]: i8, %{{.*}}: f32):
// CHECK:   %[[FP:.*]] = arith.uitofp %[[IN]] : i8 to f32
// CHECK:   %[[ZPF:.*]] = arith.sitofp %[[ZP]] : i8 to f32
// CHECK:   %[[SUB:.*]] = arith.subf %[[FP]], %[[ZPF]] : f32
// CHECK:   %[[MUL:.*]] = arith.mulf %[[SUB]], %[[SCALE]] : f32
// CHECK:   linalg.yield %[[MUL]] : f32
func.func @dequantize_affine_per_group(
    %input: !torch.vtensor<[4,16],ui8>,
    %scale: !torch.vtensor<[4,2],f32>,
    %zero_point: !torch.vtensor<[4,2],si8>)
    -> !torch.vtensor<[4,16],f32> {
  %int1 = torch.constant.int 1
  %int8 = torch.constant.int 8
  %dtype = torch.constant.int 0
  %out_dtype = torch.constant.int 6
  %none = torch.constant.none
  %block_size = torch.prim.ListConstruct %int1, %int8
      : (!torch.int, !torch.int) -> !torch.list<int>
  %out = torch.torchao.dequantize_affine
      %input, %block_size, %scale, %zero_point, %dtype, %none, %none, %out_dtype
      : !torch.vtensor<[4,16],ui8>, !torch.list<int>, !torch.vtensor<[4,2],f32>,
        !torch.vtensor<[4,2],si8>, !torch.int, !torch.none, !torch.none,
        !torch.int -> !torch.vtensor<[4,16],f32>
  return %out : !torch.vtensor<[4,16],f32>
}

// -----

// CHECK-LABEL: func.func @dequantize_affine_unsigned_zero_point(
// CHECK: ^bb0(%[[IN:.*]]: i8, %[[SCALE:.*]]: f32, %[[ZP:.*]]: i8, %{{.*}}: bf16):
// CHECK:   %[[FP:.*]] = arith.uitofp %[[IN]] : i8 to f32
// CHECK:   %[[ZPF:.*]] = arith.uitofp %[[ZP]] : i8 to f32
// CHECK:   %[[SUB:.*]] = arith.subf %[[FP]], %[[ZPF]] : f32
// CHECK:   %[[MUL:.*]] = arith.mulf %[[SUB]], %[[SCALE]] : f32
// CHECK:   arith.truncf %[[MUL]] : f32 to bf16
func.func @dequantize_affine_unsigned_zero_point(
    %input: !torch.vtensor<[4,16],ui8>,
    %scale: !torch.vtensor<[4,1],f32>,
    %zero_point: !torch.vtensor<[4,1],ui8>)
    -> !torch.vtensor<[4,16],bf16> {
  %int1 = torch.constant.int 1
  %int16 = torch.constant.int 16
  %dtype = torch.constant.int 0
  %out_dtype = torch.constant.int 15
  %none = torch.constant.none
  %block_size = torch.prim.ListConstruct %int1, %int16
      : (!torch.int, !torch.int) -> !torch.list<int>
  %out = torch.torchao.dequantize_affine
      %input, %block_size, %scale, %zero_point, %dtype, %none, %none, %out_dtype
      : !torch.vtensor<[4,16],ui8>, !torch.list<int>, !torch.vtensor<[4,1],f32>,
        !torch.vtensor<[4,1],ui8>, !torch.int, !torch.none, !torch.none,
        !torch.int -> !torch.vtensor<[4,16],bf16>
  return %out : !torch.vtensor<[4,16],bf16>
}

// -----

// CHECK: #[[IDENTITY:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[TENSOR:.*]] = affine_map<(d0, d1) -> ()>
// CHECK-LABEL: func.func @dequantize_affine_per_tensor(
// CHECK: %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[IDENTITY]], #[[TENSOR]], #[[IDENTITY]]]
// CHECK: ^bb0(%[[IN:.*]]: f8E4M3FN, %[[SCALE:.*]]: f32, %{{.*}}: bf16):
// CHECK:   %[[FP:.*]] = arith.extf %[[IN]] : f8E4M3FN to f32
// CHECK:   %[[MUL:.*]] = arith.mulf %[[FP]], %[[SCALE]] : f32
// CHECK:   arith.truncf %[[MUL]] : f32 to bf16
func.func @dequantize_affine_per_tensor(
    %input: !torch.vtensor<[?,16],f8E4M3FN>,
    %scale: !torch.vtensor<[],f32>)
    -> !torch.vtensor<[?,16],bf16> {
  %int0 = torch.constant.int 0
  %int16 = torch.constant.int 16
  %dtype = torch.constant.int 24
  %out_dtype = torch.constant.int 15
  %none = torch.constant.none
  %size = torch.aten.size.int %input, %int0
      : !torch.vtensor<[?,16],f8E4M3FN>, !torch.int -> !torch.int
  %block_size = torch.prim.ListConstruct %size, %int16
      : (!torch.int, !torch.int) -> !torch.list<int>
  %out = torch.torchao.dequantize_affine
      %input, %block_size, %scale, %none, %dtype, %none, %none, %out_dtype
      : !torch.vtensor<[?,16],f8E4M3FN>, !torch.list<int>,
        !torch.vtensor<[],f32>, !torch.none, !torch.int, !torch.none,
        !torch.none, !torch.int -> !torch.vtensor<[?,16],bf16>
  return %out : !torch.vtensor<[?,16],bf16>
}
