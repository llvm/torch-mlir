// RUN: torch-mlir-opt %s -split-input-file -verify-diagnostics -convert-torch-onnx-to-torch

// g_idx assigns arbitrary blocks to rows of K.
func.func @matmul_nbits_group_index(
    %a: !torch.vtensor<[2,32],f32>,
    %b: !torch.vtensor<[4,2,8],ui8>,
    %scales: !torch.vtensor<[4,2],f32>,
    %g_idx: !torch.vtensor<[32],si32>
    ) -> !torch.vtensor<[2,4],f32> attributes {torch.onnx_meta.opset_version = 21 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.MatMulNBits"(%a, %b, %scales, %none, %g_idx)
      {torch.onnx.K = 32 : si64, torch.onnx.N = 4 : si64, torch.onnx.block_size = 16 : si64}
      : (!torch.vtensor<[2,32],f32>, !torch.vtensor<[4,2,8],ui8>, !torch.vtensor<[4,2],f32>, !torch.none, !torch.vtensor<[32],si32>) -> !torch.vtensor<[2,4],f32>
  return %0 : !torch.vtensor<[2,4],f32>
}

// -----

// weight_prepacked selects an execution-provider specific layout of B.
func.func @matmul_nbits_weight_prepacked(
    %a: !torch.vtensor<[2,32],f32>,
    %b: !torch.vtensor<[4,2,8],ui8>,
    %scales: !torch.vtensor<[4,2],f32>
    ) -> !torch.vtensor<[2,4],f32> attributes {torch.onnx_meta.opset_version = 21 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.MatMulNBits"(%a, %b, %scales)
      {torch.onnx.K = 32 : si64, torch.onnx.N = 4 : si64, torch.onnx.block_size = 16 : si64,
       torch.onnx.weight_prepacked = 1 : si64}
      : (!torch.vtensor<[2,32],f32>, !torch.vtensor<[4,2,8],ui8>, !torch.vtensor<[4,2],f32>) -> !torch.vtensor<[2,4],f32>
  return %0 : !torch.vtensor<[2,4],f32>
}

// -----

// bits must be 2, 4 or 8. B has the shape that 8 / 3 = 2 values per byte
// gives, so only the bits check rejects the op.
func.func @matmul_nbits_3bit(
    %a: !torch.vtensor<[2,32],f32>,
    %b: !torch.vtensor<[4,2,8],ui8>,
    %scales: !torch.vtensor<[4,2],f32>
    ) -> !torch.vtensor<[2,4],f32> attributes {torch.onnx_meta.opset_version = 21 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.MatMulNBits"(%a, %b, %scales)
      {torch.onnx.K = 32 : si64, torch.onnx.N = 4 : si64, torch.onnx.bits = 3 : si64,
       torch.onnx.block_size = 16 : si64}
      : (!torch.vtensor<[2,32],f32>, !torch.vtensor<[4,2,8],ui8>, !torch.vtensor<[4,2],f32>) -> !torch.vtensor<[2,4],f32>
  return %0 : !torch.vtensor<[2,4],f32>
}

// -----

// block_size must be a power of two of at least 16.
func.func @matmul_nbits_block_size(
    %a: !torch.vtensor<[2,48],f32>,
    %b: !torch.vtensor<[4,2,12],ui8>,
    %scales: !torch.vtensor<[4,2],f32>
    ) -> !torch.vtensor<[2,4],f32> attributes {torch.onnx_meta.opset_version = 21 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.MatMulNBits"(%a, %b, %scales)
      {torch.onnx.K = 48 : si64, torch.onnx.N = 4 : si64, torch.onnx.block_size = 24 : si64}
      : (!torch.vtensor<[2,48],f32>, !torch.vtensor<[4,2,12],ui8>, !torch.vtensor<[4,2],f32>) -> !torch.vtensor<[2,4],f32>
  return %0 : !torch.vtensor<[2,4],f32>
}

// -----

// A's last dimension is K, which the attribute fixes statically.
func.func @matmul_nbits_dynamic_k(
    %a: !torch.vtensor<[2,?],f32>,
    %b: !torch.vtensor<[4,2,8],ui8>,
    %scales: !torch.vtensor<[4,2],f32>
    ) -> !torch.vtensor<[2,4],f32> attributes {torch.onnx_meta.opset_version = 21 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.MatMulNBits"(%a, %b, %scales)
      {torch.onnx.K = 32 : si64, torch.onnx.N = 4 : si64, torch.onnx.block_size = 16 : si64}
      : (!torch.vtensor<[2,?],f32>, !torch.vtensor<[4,2,8],ui8>, !torch.vtensor<[4,2],f32>) -> !torch.vtensor<[2,4],f32>
  return %0 : !torch.vtensor<[2,4],f32>
}

// -----

// B's shape must follow from K, N, bits and block_size.
func.func @matmul_nbits_weight_shape(
    %a: !torch.vtensor<[2,32],f32>,
    %b: !torch.vtensor<[4,2,16],ui8>,
    %scales: !torch.vtensor<[4,2],f32>
    ) -> !torch.vtensor<[2,4],f32> attributes {torch.onnx_meta.opset_version = 21 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.MatMulNBits"(%a, %b, %scales)
      {torch.onnx.K = 32 : si64, torch.onnx.N = 4 : si64, torch.onnx.block_size = 16 : si64}
      : (!torch.vtensor<[2,32],f32>, !torch.vtensor<[4,2,16],ui8>, !torch.vtensor<[4,2],f32>) -> !torch.vtensor<[2,4],f32>
  return %0 : !torch.vtensor<[2,4],f32>
}

// -----

// The attributes fix B's shape statically; a dynamic dimension is rejected
// rather than assumed.
func.func @matmul_nbits_dynamic_weight(
    %a: !torch.vtensor<[2,32],f32>,
    %b: !torch.vtensor<[?,2,8],ui8>,
    %scales: !torch.vtensor<[4,2],f32>
    ) -> !torch.vtensor<[2,4],f32> attributes {torch.onnx_meta.opset_version = 21 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.MatMulNBits"(%a, %b, %scales)
      {torch.onnx.K = 32 : si64, torch.onnx.N = 4 : si64, torch.onnx.block_size = 16 : si64}
      : (!torch.vtensor<[2,32],f32>, !torch.vtensor<[?,2,8],ui8>, !torch.vtensor<[4,2],f32>) -> !torch.vtensor<[2,4],f32>
  return %0 : !torch.vtensor<[2,4],f32>
}

// -----

// Zero points that are not uint8 must have A's dtype.
func.func @matmul_nbits_zero_point_dtype(
    %a: !torch.vtensor<[2,32],f16>,
    %b: !torch.vtensor<[4,2,8],ui8>,
    %scales: !torch.vtensor<[4,2],f16>,
    %zero_points: !torch.vtensor<[4,2],f32>
    ) -> !torch.vtensor<[2,4],f16> attributes {torch.onnx_meta.opset_version = 21 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.MatMulNBits"(%a, %b, %scales, %zero_points)
      {torch.onnx.K = 32 : si64, torch.onnx.N = 4 : si64, torch.onnx.block_size = 16 : si64}
      : (!torch.vtensor<[2,32],f16>, !torch.vtensor<[4,2,8],ui8>, !torch.vtensor<[4,2],f16>, !torch.vtensor<[4,2],f32>) -> !torch.vtensor<[2,4],f16>
  return %0 : !torch.vtensor<[2,4],f16>
}
