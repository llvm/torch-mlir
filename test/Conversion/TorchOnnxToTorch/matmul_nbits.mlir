// RUN: torch-mlir-opt %s -split-input-file -convert-torch-onnx-to-torch | FileCheck %s --implicit-check-not=torch.operator

// 4-bit weights without zero points use the default zero point 2^(bits - 1)
// and are unpacked low nibble first.
// CHECK-LABEL: func.func @matmul_nbits_4bit_default_zero_point(
// CHECK-SAME: %[[A:.*]]: !torch.vtensor<[2,32],f32>, %[[B:.*]]: !torch.vtensor<[4,2,8],ui8>, %[[SCALES:.*]]: !torch.vtensor<[4,2],f32>
// CHECK: %[[MINUS_ONE:.*]] = torch.constant.int -1
// CHECK: %[[EXPANDED:.*]] = torch.aten.unsqueeze %[[B]], %[[MINUS_ONE]] : !torch.vtensor<[4,2,8],ui8>, !torch.int -> !torch.vtensor<[4,2,8,1],ui8>
// CHECK: %[[SHIFTS:.*]] = torch.vtensor.literal(dense<[0, 4]> : tensor<2xui8>) : !torch.vtensor<[2],ui8>
// CHECK: %[[SHIFTED:.*]] = torch.aten.bitwise_right_shift.Tensor %[[EXPANDED]], %[[SHIFTS]] : {{.*}} -> !torch.vtensor<[4,2,8,2],ui8>
// CHECK: %[[MASK:.*]] = torch.constant.int 15
// CHECK: %[[VALUES:.*]] = torch.aten.bitwise_and.Scalar %[[SHIFTED]], %[[MASK]] : {{.*}} -> !torch.vtensor<[4,2,8,2],ui8>
// CHECK: %[[BLOCKS:.*]] = torch.aten.flatten.using_ints %[[VALUES]], {{.*}} -> !torch.vtensor<[4,2,16],ui8>
// CHECK: %[[Q:.*]] = torch.aten.flatten.using_ints %[[BLOCKS]], {{.*}} -> !torch.vtensor<[4,32],ui8>
// CHECK: %[[ZP:.*]] = torch.vtensor.literal(dense<8> : tensor<4x2xui8>) : !torch.vtensor<[4,2],ui8>
// CHECK: %[[QMIN:.*]] = torch.constant.int 0
// CHECK: %[[QMAX:.*]] = torch.constant.int 15
// CHECK: %[[UINT8:.*]] = torch.constant.int 0
// CHECK: %[[GROUP:.*]] = torch.constant.int 16
// CHECK: %[[F32:.*]] = torch.constant.int 6
// CHECK: %[[W:.*]] = torch.quantized_decomposed.dequantize_per_channel_group %[[Q]], %[[SCALES]], %[[ZP]], %[[QMIN]], %[[QMAX]], %[[UINT8]], %[[GROUP]], %[[F32]]
// CHECK-SAME: -> !torch.vtensor<[4,32],f32>
// CHECK: %[[NONE:.*]] = torch.constant.none
// CHECK: %[[Y:.*]] = torch.aten.linear %[[A]], %[[W]], %[[NONE]] : {{.*}} -> !torch.vtensor<[2,4],f32>
// CHECK: return %[[Y]]
func.func @matmul_nbits_4bit_default_zero_point(
    %a: !torch.vtensor<[2,32],f32>,
    %b: !torch.vtensor<[4,2,8],ui8>,
    %scales: !torch.vtensor<[4,2],f32>
    ) -> !torch.vtensor<[2,4],f32>
    attributes {torch.onnx_meta.opset_version = 21 : si64,
                torch.onnx_meta.opset_versions = {"com.microsoft" = 1 : si64}} {
  %0 = torch.operator "onnx.MatMulNBits"(%a, %b, %scales)
      {torch.onnx.K = 32 : si64, torch.onnx.N = 4 : si64,
       torch.onnx.bits = 4 : si64, torch.onnx.block_size = 16 : si64}
      : (!torch.vtensor<[2,32],f32>, !torch.vtensor<[4,2,8],ui8>, !torch.vtensor<[4,2],f32>) -> !torch.vtensor<[2,4],f32>
  return %0 : !torch.vtensor<[2,4],f32>
}

// -----

// 8-bit weights need no unpacking; uint8 zero points and the bias apply as
// given, and A may carry batch dimensions.
// CHECK-LABEL: func.func @matmul_nbits_8bit_zero_point_bias(
// CHECK-SAME: %[[A:.*]]: !torch.vtensor<[2,3,32],f32>, %[[B:.*]]: !torch.vtensor<[4,2,16],ui8>, %[[SCALES:.*]]: !torch.vtensor<[4,2],f32>, %[[ZP:.*]]: !torch.vtensor<[4,2],ui8>, %[[BIAS:.*]]: !torch.vtensor<[4],f32>
// CHECK-NOT: torch.aten.bitwise_right_shift.Tensor
// CHECK: %[[Q:.*]] = torch.aten.flatten.using_ints %[[B]], {{.*}} -> !torch.vtensor<[4,32],ui8>
// CHECK: %[[QMIN:.*]] = torch.constant.int 0
// CHECK: %[[QMAX:.*]] = torch.constant.int 255
// CHECK: %[[W:.*]] = torch.quantized_decomposed.dequantize_per_channel_group %[[Q]], %[[SCALES]], %[[ZP]], %[[QMIN]], %[[QMAX]]
// CHECK-SAME: -> !torch.vtensor<[4,32],f32>
// CHECK: %[[Y:.*]] = torch.aten.linear %[[A]], %[[W]], %[[BIAS]] : {{.*}} -> !torch.vtensor<[2,3,4],f32>
// CHECK: return %[[Y]]
func.func @matmul_nbits_8bit_zero_point_bias(
    %a: !torch.vtensor<[2,3,32],f32>,
    %b: !torch.vtensor<[4,2,16],ui8>,
    %scales: !torch.vtensor<[4,2],f32>,
    %zero_points: !torch.vtensor<[4,2],ui8>,
    %bias: !torch.vtensor<[4],f32>
    ) -> !torch.vtensor<[2,3,4],f32>
    attributes {torch.onnx_meta.opset_version = 21 : si64,
                torch.onnx_meta.opset_versions = {"com.microsoft" = 1 : si64}} {
  %none = torch.constant.none
  %0 = torch.operator "onnx.MatMulNBits"(%a, %b, %scales, %zero_points, %none, %bias)
      {torch.onnx.K = 32 : si64, torch.onnx.N = 4 : si64,
       torch.onnx.bits = 8 : si64, torch.onnx.block_size = 16 : si64}
      : (!torch.vtensor<[2,3,32],f32>, !torch.vtensor<[4,2,16],ui8>, !torch.vtensor<[4,2],f32>, !torch.vtensor<[4,2],ui8>, !torch.none, !torch.vtensor<[4],f32>) -> !torch.vtensor<[2,3,4],f32>
  return %0 : !torch.vtensor<[2,3,4],f32>
}

// -----

// K = 40 is not a multiple of block_size = 16: B holds three blocks, so the
// dequantized weight spans the padded K = 48 and A is zero padded to match.
// The 2-bit packed zero points of the three blocks occupy one padded byte per
// row, and the flattened scales are reshaped to [N, k_blocks].
// CHECK-LABEL: func.func @matmul_nbits_2bit_ragged_k(
// CHECK-SAME: %[[A:.*]]: !torch.vtensor<[2,40],f32>, %[[B:.*]]: !torch.vtensor<[4,3,4],ui8>, %[[SCALES:.*]]: !torch.vtensor<[12],f32>, %[[ZP:.*]]: !torch.vtensor<[4,1],ui8>
// CHECK: %[[EXPANDED:.*]] = torch.aten.unsqueeze %[[B]], {{.*}} -> !torch.vtensor<[4,3,4,1],ui8>
// CHECK: %[[SHIFTS:.*]] = torch.vtensor.literal(dense<[0, 2, 4, 6]> : tensor<4xui8>) : !torch.vtensor<[4],ui8>
// CHECK: %[[SHIFTED:.*]] = torch.aten.bitwise_right_shift.Tensor %[[EXPANDED]], %[[SHIFTS]] : {{.*}} -> !torch.vtensor<[4,3,4,4],ui8>
// CHECK: %[[MASK:.*]] = torch.constant.int 3
// CHECK: %[[VALUES:.*]] = torch.aten.bitwise_and.Scalar %[[SHIFTED]], %[[MASK]] : {{.*}} -> !torch.vtensor<[4,3,4,4],ui8>
// CHECK: %[[BLOCKS:.*]] = torch.aten.flatten.using_ints %[[VALUES]], {{.*}} -> !torch.vtensor<[4,3,16],ui8>
// CHECK: %[[ONE:.*]] = torch.constant.int 1
// CHECK: %[[Q:.*]] = torch.aten.flatten.using_ints %[[BLOCKS]], %[[ONE]], {{.*}} -> !torch.vtensor<[4,48],ui8>
// CHECK: %[[SCALES_2D:.*]] = torch.aten.reshape %[[SCALES]], {{.*}} -> !torch.vtensor<[4,3],f32>
// CHECK: %[[ZP_EXPANDED:.*]] = torch.aten.unsqueeze %[[ZP]], {{.*}} -> !torch.vtensor<[4,1,1],ui8>
// CHECK: %[[ZP_SHIFTS:.*]] = torch.vtensor.literal(dense<[0, 2, 4, 6]> : tensor<4xui8>) : !torch.vtensor<[4],ui8>
// CHECK: %[[ZP_SHIFTED:.*]] = torch.aten.bitwise_right_shift.Tensor %[[ZP_EXPANDED]], %[[ZP_SHIFTS]] : {{.*}} -> !torch.vtensor<[4,1,4],ui8>
// CHECK: %[[ZP_MASK:.*]] = torch.constant.int 3
// CHECK: %[[ZP_UNPACKED:.*]] = torch.aten.bitwise_and.Scalar %[[ZP_SHIFTED]], %[[ZP_MASK]] : {{.*}} -> !torch.vtensor<[4,1,4],ui8>
// CHECK: %[[ZP_VALUES:.*]] = torch.aten.flatten.using_ints %[[ZP_UNPACKED]], {{.*}} -> !torch.vtensor<[4,4],ui8>
// CHECK: %[[ZP_START:.*]] = torch.constant.int 0
// CHECK: %[[ZP_END:.*]] = torch.constant.int 3
// CHECK: %[[ZP_BLOCKS:.*]] = torch.aten.slice.Tensor %[[ZP_VALUES]], %[[ONE]], %[[ZP_START]], %[[ZP_END]], %[[ONE]] : {{.*}} -> !torch.vtensor<[4,3],ui8>
// CHECK: %[[QMAX:.*]] = torch.constant.int 3
// CHECK: %[[W:.*]] = torch.quantized_decomposed.dequantize_per_channel_group %[[Q]], %[[SCALES_2D]], %[[ZP_BLOCKS]], %{{.*}}, %[[QMAX]]
// CHECK-SAME: -> !torch.vtensor<[4,48],f32>
// CHECK: %[[PAD_BEFORE:.*]] = torch.constant.int 0
// CHECK: %[[PAD_AFTER:.*]] = torch.constant.int 8
// CHECK: %[[PADDING:.*]] = torch.prim.ListConstruct %[[PAD_BEFORE]], %[[PAD_AFTER]]
// CHECK: %[[PAD_VALUE:.*]] = torch.constant.float 0.000000e+00
// CHECK: %[[A_PADDED:.*]] = torch.aten.constant_pad_nd %[[A]], %[[PADDING]], %[[PAD_VALUE]] : {{.*}} -> !torch.vtensor<[2,48],f32>
// CHECK-NOT: torch.aten.slice.Tensor
// CHECK: %[[Y:.*]] = torch.aten.linear %[[A_PADDED]], %[[W]], %{{.*}} : !torch.vtensor<[2,48],f32>, !torch.vtensor<[4,48],f32>, !torch.none -> !torch.vtensor<[2,4],f32>
func.func @matmul_nbits_2bit_ragged_k(
    %a: !torch.vtensor<[2,40],f32>,
    %b: !torch.vtensor<[4,3,4],ui8>,
    %scales: !torch.vtensor<[12],f32>,
    %zero_points: !torch.vtensor<[4,1],ui8>
    ) -> !torch.vtensor<[2,4],f32>
    attributes {torch.onnx_meta.opset_version = 21 : si64,
                torch.onnx_meta.opset_versions = {"com.microsoft" = 1 : si64}} {
  %0 = torch.operator "onnx.MatMulNBits"(%a, %b, %scales, %zero_points)
      {torch.onnx.K = 40 : si64, torch.onnx.N = 4 : si64,
       torch.onnx.bits = 2 : si64, torch.onnx.block_size = 16 : si64}
      : (!torch.vtensor<[2,40],f32>, !torch.vtensor<[4,3,4],ui8>, !torch.vtensor<[12],f32>, !torch.vtensor<[4,1],ui8>) -> !torch.vtensor<[2,4],f32>
  return %0 : !torch.vtensor<[2,4],f32>
}

// -----

// Floating-point zero points, as HQQ produces, reach the dequantize without a
// cast, so their fractional part is kept. The flattened [N * k_blocks] form is
// reshaped to [N, k_blocks].
// CHECK-LABEL: func.func @matmul_nbits_float_zero_point(
// CHECK-SAME: %[[A:.*]]: !torch.vtensor<[2,32],f16>, %{{.*}}: !torch.vtensor<[4,2,8],ui8>, %[[SCALES:.*]]: !torch.vtensor<[4,2],f16>, %[[ZP:.*]]: !torch.vtensor<[8],f16>
// CHECK: %[[Q:.*]] = torch.aten.flatten.using_ints {{.*}} -> !torch.vtensor<[4,32],ui8>
// CHECK: %[[ZP_2D:.*]] = torch.aten.reshape %[[ZP]], {{.*}} -> !torch.vtensor<[4,2],f16>
// CHECK: %[[F16:.*]] = torch.constant.int 5
// CHECK: %[[W:.*]] = torch.quantized_decomposed.dequantize_per_channel_group %[[Q]], %[[SCALES]], %[[ZP_2D]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[F16]]
// CHECK-SAME: -> !torch.vtensor<[4,32],f16>
// CHECK: torch.aten.linear %[[A]], %[[W]], %{{.*}} : {{.*}} -> !torch.vtensor<[2,4],f16>
func.func @matmul_nbits_float_zero_point(
    %a: !torch.vtensor<[2,32],f16>,
    %b: !torch.vtensor<[4,2,8],ui8>,
    %scales: !torch.vtensor<[4,2],f16>,
    %zero_points: !torch.vtensor<[8],f16>
    ) -> !torch.vtensor<[2,4],f16>
    attributes {torch.onnx_meta.opset_version = 21 : si64,
                torch.onnx_meta.opset_versions = {"com.microsoft" = 1 : si64}} {
  %0 = torch.operator "onnx.MatMulNBits"(%a, %b, %scales, %zero_points)
      {torch.onnx.K = 32 : si64, torch.onnx.N = 4 : si64,
       torch.onnx.bits = 4 : si64, torch.onnx.block_size = 16 : si64}
      : (!torch.vtensor<[2,32],f16>, !torch.vtensor<[4,2,8],ui8>, !torch.vtensor<[4,2],f16>, !torch.vtensor<[8],f16>) -> !torch.vtensor<[2,4],f16>
  return %0 : !torch.vtensor<[2,4],f16>
}

// -----

// accuracy_level 4 quantizes the zero-padded A dynamically to int8 per block
// of block_size along K.
// CHECK-LABEL: func.func @matmul_nbits_accuracy_level_4(
// CHECK-SAME: %[[A:.*]]: !torch.vtensor<[?,40],f32>, %[[B:.*]]: !torch.vtensor<[4,3,8],ui8>, %[[SCALES:.*]]: !torch.vtensor<[4,3],f32>
// CHECK: %[[Q:.*]] = torch.aten.flatten.using_ints {{.*}} -> !torch.vtensor<[4,48],ui8>
// CHECK: %[[ZP:.*]] = torch.vtensor.literal(dense<8> : tensor<4x3xui8>) : !torch.vtensor<[4,3],ui8>
// CHECK: %[[W:.*]] = torch.quantized_decomposed.dequantize_per_channel_group %[[Q]], %[[SCALES]], %[[ZP]],
// CHECK-SAME: -> !torch.vtensor<[4,48],f32>
// CHECK: %[[PAD_BEFORE:.*]] = torch.constant.int 0
// CHECK: %[[PAD_AFTER:.*]] = torch.constant.int 8
// CHECK: %[[PADDING:.*]] = torch.prim.ListConstruct %[[PAD_BEFORE]], %[[PAD_AFTER]]
// CHECK: %[[PAD_VALUE:.*]] = torch.constant.float 0.000000e+00
// CHECK: %[[A_PADDED:.*]] = torch.aten.constant_pad_nd %[[A]], %[[PADDING]], %[[PAD_VALUE]] : {{.*}} -> !torch.vtensor<[?,48],f32>
// CHECK: %[[A_BLOCKS:.*]] = torch.aten.unflatten.int %[[A_PADDED]], {{.*}} -> !torch.vtensor<[?,3,16],f32>
// CHECK: %[[INT8:.*]] = torch.constant.int 1
// CHECK: %[[SCALE:.*]], %[[ZERO_POINT:.*]] = torch.quantized_decomposed.choose_qparams_per_token %[[A_BLOCKS]], %[[INT8]] : !torch.vtensor<[?,3,16],f32>, !torch.int -> !torch.vtensor<[?,3,1],f32>, !torch.vtensor<[?,3,1],si64>
// CHECK: %[[QMIN:.*]] = torch.constant.int -128
// CHECK: %[[QMAX:.*]] = torch.constant.int 127
// CHECK: %[[A_Q:.*]] = torch.quantized_decomposed.quantize_per_token %[[A_BLOCKS]], %[[SCALE]], %[[ZERO_POINT]], %[[QMIN]], %[[QMAX]], %[[INT8]] : {{.*}} -> !torch.vtensor<[?,3,16],si8>
// CHECK: %[[F32:.*]] = torch.constant.int 6
// CHECK: %[[A_DQ:.*]] = torch.quantized_decomposed.dequantize_per_token %[[A_Q]], %[[SCALE]], %[[ZERO_POINT]], %[[QMIN]], %[[QMAX]], %[[INT8]], %[[F32]] : {{.*}} -> !torch.vtensor<[?,3,16],f32>
// CHECK: %[[A_FLAT:.*]] = torch.aten.flatten.using_ints %[[A_DQ]], {{.*}} -> !torch.vtensor<[?,48],f32>
// CHECK-NOT: torch.aten.slice.Tensor
// CHECK: %[[Y:.*]] = torch.aten.linear %[[A_FLAT]], %[[W]], %{{.*}} : !torch.vtensor<[?,48],f32>, !torch.vtensor<[4,48],f32>, !torch.none -> !torch.vtensor<[?,4],f32>
func.func @matmul_nbits_accuracy_level_4(
    %a: !torch.vtensor<[?,40],f32>,
    %b: !torch.vtensor<[4,3,8],ui8>,
    %scales: !torch.vtensor<[4,3],f32>
    ) -> !torch.vtensor<[?,4],f32>
    attributes {torch.onnx_meta.opset_version = 21 : si64,
                torch.onnx_meta.opset_versions = {"com.microsoft" = 1 : si64}} {
  %0 = torch.operator "onnx.MatMulNBits"(%a, %b, %scales)
      {torch.onnx.K = 40 : si64, torch.onnx.N = 4 : si64,
       torch.onnx.bits = 4 : si64, torch.onnx.block_size = 16 : si64,
       torch.onnx.accuracy_level = 4 : si64}
      : (!torch.vtensor<[?,40],f32>, !torch.vtensor<[4,3,8],ui8>, !torch.vtensor<[4,3],f32>) -> !torch.vtensor<[?,4],f32>
  return %0 : !torch.vtensor<[?,4],f32>
}

// -----

// onnxruntime has no int8 kernel for floating-point zero points and keeps A
// in floating point at accuracy_level 4; so does the lowering.
// CHECK-LABEL: func.func @matmul_nbits_accuracy_level_4_float_zero_point(
// CHECK-SAME: %[[A:[^:]*]]: !torch.vtensor<[2,32],f32>
// CHECK: %[[W:.*]] = torch.quantized_decomposed.dequantize_per_channel_group
// CHECK-NOT: torch.quantized_decomposed.{{.*}}_per_token
// CHECK: torch.aten.linear %[[A]], %[[W]], %{{.*}} : {{.*}} -> !torch.vtensor<[2,4],f32>
func.func @matmul_nbits_accuracy_level_4_float_zero_point(
    %a: !torch.vtensor<[2,32],f32>,
    %b: !torch.vtensor<[4,2,8],ui8>,
    %scales: !torch.vtensor<[4,2],f32>,
    %zero_points: !torch.vtensor<[4,2],f32>
    ) -> !torch.vtensor<[2,4],f32>
    attributes {torch.onnx_meta.opset_version = 21 : si64,
                torch.onnx_meta.opset_versions = {"com.microsoft" = 1 : si64}} {
  %0 = torch.operator "onnx.MatMulNBits"(%a, %b, %scales, %zero_points)
      {torch.onnx.K = 32 : si64, torch.onnx.N = 4 : si64,
       torch.onnx.bits = 4 : si64, torch.onnx.block_size = 16 : si64,
       torch.onnx.accuracy_level = 4 : si64}
      : (!torch.vtensor<[2,32],f32>, !torch.vtensor<[4,2,8],ui8>, !torch.vtensor<[4,2],f32>, !torch.vtensor<[4,2],f32>) -> !torch.vtensor<[2,4],f32>
  return %0 : !torch.vtensor<[2,4],f32>
}
