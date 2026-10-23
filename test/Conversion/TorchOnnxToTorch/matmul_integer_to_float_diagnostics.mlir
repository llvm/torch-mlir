// RUN: torch-mlir-opt %s -split-input-file -verify-diagnostics -convert-torch-onnx-to-torch

// unsupported input dtype.
func.func @matmul_integer_to_float_unsupported_input_dtype(
    %a: !torch.vtensor<[4,3],si16>,
    %b: !torch.vtensor<[3,5],ui8>,
    %as: !torch.vtensor<[],f32>,
    %bs: !torch.vtensor<[],f32>
    ) -> !torch.vtensor<[4,5],f32> attributes {torch.onnx_meta.opset_version = 17 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %as, %bs)
      : (!torch.vtensor<[4,3],si16>, !torch.vtensor<[3,5],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[4,5],f32>
  return %0 : !torch.vtensor<[4,5],f32>
}

// -----

// unsupported b dtype.
func.func @matmul_integer_to_float_unsupported_b_dtype(
    %a: !torch.vtensor<[4,3],si8>,
    %b: !torch.vtensor<[3,5],f32>,
    %as: !torch.vtensor<[],f32>,
    %bs: !torch.vtensor<[],f32>
    ) -> !torch.vtensor<[4,5],f32> attributes {torch.onnx_meta.opset_version = 17 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %as, %bs)
      : (!torch.vtensor<[4,3],si8>, !torch.vtensor<[3,5],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[4,5],f32>
  return %0 : !torch.vtensor<[4,5],f32>
}

// -----

// scalar input.
func.func @matmul_integer_to_float_scalar_input(
    %a: !torch.vtensor<[],si8>,
    %b: !torch.vtensor<[3,5],ui8>,
    %as: !torch.vtensor<[],f32>,
    %bs: !torch.vtensor<[],f32>
    ) -> !torch.vtensor<[4,5],f32> attributes {torch.onnx_meta.opset_version = 17 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %as, %bs)
      : (!torch.vtensor<[],si8>, !torch.vtensor<[3,5],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[4,5],f32>
  return %0 : !torch.vtensor<[4,5],f32>
}

// -----

// unranked input.
func.func @matmul_integer_to_float_unranked_input(
    %a: !torch.vtensor<*,si8>,
    %b: !torch.vtensor<[3,5],ui8>,
    %as: !torch.vtensor<[],f32>,
    %bs: !torch.vtensor<[],f32>
    ) -> !torch.vtensor<[4,5],f32> attributes {torch.onnx_meta.opset_version = 17 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %as, %bs)
      : (!torch.vtensor<*,si8>, !torch.vtensor<[3,5],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[4,5],f32>
  return %0 : !torch.vtensor<[4,5],f32>
}

// -----

// scale dtype.
func.func @matmul_integer_to_float_scale_dtype(
    %a: !torch.vtensor<[4,3],si8>,
    %b: !torch.vtensor<[3,5],ui8>,
    %as: !torch.vtensor<[],f16>,
    %bs: !torch.vtensor<[],f32>
    ) -> !torch.vtensor<[4,5],f32> attributes {torch.onnx_meta.opset_version = 17 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %as, %bs)
      : (!torch.vtensor<[4,3],si8>, !torch.vtensor<[3,5],ui8>, !torch.vtensor<[],f16>, !torch.vtensor<[],f32>) -> !torch.vtensor<[4,5],f32>
  return %0 : !torch.vtensor<[4,5],f32>
}

// -----

// scale rank.
func.func @matmul_integer_to_float_scale_rank(
    %a: !torch.vtensor<[4,3],si8>,
    %b: !torch.vtensor<[3,5],ui8>,
    %as: !torch.vtensor<[1,3],f32>,
    %bs: !torch.vtensor<[],f32>
    ) -> !torch.vtensor<[4,5],f32> attributes {torch.onnx_meta.opset_version = 17 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %as, %bs)
      : (!torch.vtensor<[4,3],si8>, !torch.vtensor<[3,5],ui8>, !torch.vtensor<[1,3],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[4,5],f32>
  return %0 : !torch.vtensor<[4,5],f32>
}

// -----

// a scale length.
func.func @matmul_integer_to_float_a_scale_length(
    %a: !torch.vtensor<[4,3],si8>,
    %b: !torch.vtensor<[3,5],ui8>,
    %as: !torch.vtensor<[4],f32>,
    %bs: !torch.vtensor<[],f32>
    ) -> !torch.vtensor<[4,5],f32> attributes {torch.onnx_meta.opset_version = 17 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %as, %bs)
      : (!torch.vtensor<[4,3],si8>, !torch.vtensor<[3,5],ui8>, !torch.vtensor<[4],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[4,5],f32>
  return %0 : !torch.vtensor<[4,5],f32>
}

// -----

// b scale length.
func.func @matmul_integer_to_float_b_scale_length(
    %a: !torch.vtensor<[4,3],si8>,
    %b: !torch.vtensor<[3,5],ui8>,
    %as: !torch.vtensor<[],f32>,
    %bs: !torch.vtensor<[3],f32>
    ) -> !torch.vtensor<[4,5],f32> attributes {torch.onnx_meta.opset_version = 17 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %as, %bs)
      : (!torch.vtensor<[4,3],si8>, !torch.vtensor<[3,5],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[3],f32>) -> !torch.vtensor<[4,5],f32>
  return %0 : !torch.vtensor<[4,5],f32>
}

// -----

// zero point dtype.
func.func @matmul_integer_to_float_zero_point_dtype(
    %a: !torch.vtensor<[4,3],si8>,
    %b: !torch.vtensor<[3,5],ui8>,
    %as: !torch.vtensor<[],f32>,
    %bs: !torch.vtensor<[],f32>,
    %az: !torch.vtensor<[],ui8>
    ) -> !torch.vtensor<[4,5],f32> attributes {torch.onnx_meta.opset_version = 17 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %as, %bs, %az)
      : (!torch.vtensor<[4,3],si8>, !torch.vtensor<[3,5],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>) -> !torch.vtensor<[4,5],f32>
  return %0 : !torch.vtensor<[4,5],f32>
}

// -----

// zero point rank.
func.func @matmul_integer_to_float_zero_point_rank(
    %a: !torch.vtensor<[4,3],si8>,
    %b: !torch.vtensor<[3,5],ui8>,
    %as: !torch.vtensor<[],f32>,
    %bs: !torch.vtensor<[],f32>,
    %bz: !torch.vtensor<[1,5],ui8>
    ) -> !torch.vtensor<[4,5],f32> attributes {torch.onnx_meta.opset_version = 17 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %as, %bs, %none, %bz)
      : (!torch.vtensor<[4,3],si8>, !torch.vtensor<[3,5],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.none, !torch.vtensor<[1,5],ui8>) -> !torch.vtensor<[4,5],f32>
  return %0 : !torch.vtensor<[4,5],f32>
}

// -----

// zero point length.
func.func @matmul_integer_to_float_zero_point_length(
    %a: !torch.vtensor<[4,3],si8>,
    %b: !torch.vtensor<[3,5],ui8>,
    %as: !torch.vtensor<[],f32>,
    %bs: !torch.vtensor<[],f32>,
    %az: !torch.vtensor<[4],si8>
    ) -> !torch.vtensor<[4,5],f32> attributes {torch.onnx_meta.opset_version = 17 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %as, %bs, %az)
      : (!torch.vtensor<[4,3],si8>, !torch.vtensor<[3,5],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[4],si8>) -> !torch.vtensor<[4,5],f32>
  return %0 : !torch.vtensor<[4,5],f32>
}

// -----

// qparam lengths mismatch.
func.func @matmul_integer_to_float_qparam_lengths_mismatch(
    %a: !torch.vtensor<[4,?],si8>,
    %b: !torch.vtensor<[3,5],ui8>,
    %as: !torch.vtensor<[3],f32>,
    %bs: !torch.vtensor<[],f32>,
    %az: !torch.vtensor<[4],si8>
    ) -> !torch.vtensor<[4,5],f32> attributes {torch.onnx_meta.opset_version = 17 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %as, %bs, %az)
      : (!torch.vtensor<[4,?],si8>, !torch.vtensor<[3,5],ui8>, !torch.vtensor<[3],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[4],si8>) -> !torch.vtensor<[4,5],f32>
  return %0 : !torch.vtensor<[4,5],f32>
}

// -----

// bias dtype.
func.func @matmul_integer_to_float_bias_dtype(
    %a: !torch.vtensor<[4,3],si8>,
    %b: !torch.vtensor<[3,5],ui8>,
    %as: !torch.vtensor<[],f32>,
    %bs: !torch.vtensor<[],f32>,
    %bias: !torch.vtensor<[5],si32>
    ) -> !torch.vtensor<[4,5],f32> attributes {torch.onnx_meta.opset_version = 17 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %as, %bs, %none, %none, %bias)
      : (!torch.vtensor<[4,3],si8>, !torch.vtensor<[3,5],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.vtensor<[5],si32>) -> !torch.vtensor<[4,5],f32>
  return %0 : !torch.vtensor<[4,5],f32>
}

// -----

// bias rank.
func.func @matmul_integer_to_float_bias_rank(
    %a: !torch.vtensor<[4,3],si8>,
    %b: !torch.vtensor<[3,5],ui8>,
    %as: !torch.vtensor<[],f32>,
    %bs: !torch.vtensor<[],f32>,
    %bias: !torch.vtensor<[1,5],f32>
    ) -> !torch.vtensor<[4,5],f32> attributes {torch.onnx_meta.opset_version = 17 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %as, %bs, %none, %none, %bias)
      : (!torch.vtensor<[4,3],si8>, !torch.vtensor<[3,5],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.vtensor<[1,5],f32>) -> !torch.vtensor<[4,5],f32>
  return %0 : !torch.vtensor<[4,5],f32>
}

// -----

// bias length.
func.func @matmul_integer_to_float_bias_length(
    %a: !torch.vtensor<[4,3],si8>,
    %b: !torch.vtensor<[3,5],ui8>,
    %as: !torch.vtensor<[],f32>,
    %bs: !torch.vtensor<[],f32>,
    %bias: !torch.vtensor<[3],f32>
    ) -> !torch.vtensor<[4,5],f32> attributes {torch.onnx_meta.opset_version = 17 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %as, %bs, %none, %none, %bias)
      : (!torch.vtensor<[4,3],si8>, !torch.vtensor<[3,5],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.vtensor<[3],f32>) -> !torch.vtensor<[4,5],f32>
  return %0 : !torch.vtensor<[4,5],f32>
}

// -----

// result dtype.
func.func @matmul_integer_to_float_result_dtype(
    %a: !torch.vtensor<[4,3],si8>,
    %b: !torch.vtensor<[3,5],ui8>,
    %as: !torch.vtensor<[],f32>,
    %bs: !torch.vtensor<[],f32>
    ) -> !torch.vtensor<[4,5],f16> attributes {torch.onnx_meta.opset_version = 17 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %as, %bs)
      : (!torch.vtensor<[4,3],si8>, !torch.vtensor<[3,5],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[4,5],f16>
  return %0 : !torch.vtensor<[4,5],f16>
}

// -----

// required none.
func.func @matmul_integer_to_float_required_none(
    %a: !torch.vtensor<[4,3],si8>,
    %b: !torch.vtensor<[3,5],ui8>,
    %as: !torch.none,
    %bs: !torch.vtensor<[],f32>
    ) -> !torch.vtensor<[4,5],f32> attributes {torch.onnx_meta.opset_version = 17 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %as, %bs)
      : (!torch.vtensor<[4,3],si8>, !torch.vtensor<[3,5],ui8>, !torch.none, !torch.vtensor<[],f32>) -> !torch.vtensor<[4,5],f32>
  return %0 : !torch.vtensor<[4,5],f32>
}

// -----

// optional non tensor.
func.func @matmul_integer_to_float_optional_non_tensor(
    %a: !torch.vtensor<[4,3],si8>,
    %b: !torch.vtensor<[3,5],ui8>,
    %as: !torch.vtensor<[],f32>,
    %bs: !torch.vtensor<[],f32>,
    %az: !torch.int
    ) -> !torch.vtensor<[4,5],f32> attributes {torch.onnx_meta.opset_version = 17 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %as, %bs, %az)
      : (!torch.vtensor<[4,3],si8>, !torch.vtensor<[3,5],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.int) -> !torch.vtensor<[4,5],f32>
  return %0 : !torch.vtensor<[4,5],f32>
}

// -----

// too few inputs.
func.func @matmul_integer_to_float_too_few_inputs(
    %a: !torch.vtensor<[4,3],si8>,
    %b: !torch.vtensor<[3,5],ui8>,
    %as: !torch.vtensor<[],f32>
    ) -> !torch.vtensor<[4,5],f32> attributes {torch.onnx_meta.opset_version = 17 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %as)
      : (!torch.vtensor<[4,3],si8>, !torch.vtensor<[3,5],ui8>, !torch.vtensor<[],f32>) -> !torch.vtensor<[4,5],f32>
  return %0 : !torch.vtensor<[4,5],f32>
}

// -----

// Unknown qparam lengths require an unknown input column count.
func.func @matmul_integer_to_float_unknown_a_scale_length(
    %a: !torch.vtensor<[4,3],si8>,
    %b: !torch.vtensor<[3,5],ui8>,
    %as: !torch.vtensor<[?],f32>,
    %bs: !torch.vtensor<[],f32>
    ) -> !torch.vtensor<[4,5],f32> attributes {torch.onnx_meta.opset_version = 17 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %as, %bs)
      : (!torch.vtensor<[4,3],si8>, !torch.vtensor<[3,5],ui8>, !torch.vtensor<[?],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[4,5],f32>
  return %0 : !torch.vtensor<[4,5],f32>
}

// -----

// Unknown qparam lengths require an unknown input column count.
func.func @matmul_integer_to_float_unknown_b_scale_length(
    %a: !torch.vtensor<[4,?],si8>,
    %b: !torch.vtensor<[?,5],ui8>,
    %as: !torch.vtensor<[],f32>,
    %bs: !torch.vtensor<[?],f32>
    ) -> !torch.vtensor<[4,5],f32> attributes {torch.onnx_meta.opset_version = 17 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %as, %bs)
      : (!torch.vtensor<[4,?],si8>, !torch.vtensor<[?,5],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[?],f32>) -> !torch.vtensor<[4,5],f32>
  return %0 : !torch.vtensor<[4,5],f32>
}

// -----

// Unknown qparam lengths require an unknown input column count.
func.func @matmul_integer_to_float_unknown_a_zero_point_length(
    %a: !torch.vtensor<[4,3],si8>,
    %b: !torch.vtensor<[3,5],ui8>,
    %as: !torch.vtensor<[],f32>,
    %bs: !torch.vtensor<[],f32>,
    %az: !torch.vtensor<[?],si8>
    ) -> !torch.vtensor<[4,5],f32> attributes {torch.onnx_meta.opset_version = 17 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %as, %bs, %az)
      : (!torch.vtensor<[4,3],si8>, !torch.vtensor<[3,5],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[?],si8>) -> !torch.vtensor<[4,5],f32>
  return %0 : !torch.vtensor<[4,5],f32>
}

// -----

// Unknown qparam lengths require an unknown input column count.
func.func @matmul_integer_to_float_unknown_b_zero_point_length(
    %a: !torch.vtensor<[4,?],si8>,
    %b: !torch.vtensor<[?,5],ui8>,
    %as: !torch.vtensor<[],f32>,
    %bs: !torch.vtensor<[5],f32>,
    %bz: !torch.vtensor<[?],ui8>
    ) -> !torch.vtensor<[4,5],f32> attributes {torch.onnx_meta.opset_version = 17 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %as, %bs, %none, %bz)
      : (!torch.vtensor<[4,?],si8>, !torch.vtensor<[?,5],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[5],f32>, !torch.none, !torch.vtensor<[?],ui8>) -> !torch.vtensor<[4,5],f32>
  return %0 : !torch.vtensor<[4,5],f32>
}

// -----

// too many inputs.
func.func @matmul_integer_to_float_too_many_inputs(
    %a: !torch.vtensor<[4,3],si8>,
    %b: !torch.vtensor<[3,5],ui8>,
    %as: !torch.vtensor<[],f32>,
    %bs: !torch.vtensor<[],f32>
    ) -> !torch.vtensor<[4,5],f32> attributes {torch.onnx_meta.opset_version = 17 : si64} {
  %none = torch.constant.none
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%a, %b, %as, %bs, %none, %none, %none, %none)
      : (!torch.vtensor<[4,3],si8>, !torch.vtensor<[3,5],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.none, !torch.none) -> !torch.vtensor<[4,5],f32>
  return %0 : !torch.vtensor<[4,5],f32>
}
