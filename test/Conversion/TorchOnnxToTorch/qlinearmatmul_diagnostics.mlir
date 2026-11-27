// RUN: torch-mlir-opt %s -split-input-file -verify-diagnostics -convert-torch-onnx-to-torch

// A 1-D `a` has no row dimension, so only per-tensor quantization parameters
// are allowed for it.
func.func @test_qlinearmatmul_a_1d_non_per_tensor(
    %arg0: !torch.vtensor<[4],ui8>, %arg1: !torch.vtensor<[4],f32>,
    %arg2: !torch.vtensor<[4],ui8>, %arg3: !torch.vtensor<[4,3],ui8>,
    %arg4: !torch.vtensor<[],f32>, %arg5: !torch.vtensor<[],ui8>,
    %arg6: !torch.vtensor<[],f32>, %arg7: !torch.vtensor<[],ui8>)
    -> !torch.vtensor<[3],ui8>
    attributes {torch.onnx_meta.opset_version = 21 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.QLinearMatMul"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7)
      : (!torch.vtensor<[4],ui8>, !torch.vtensor<[4],f32>, !torch.vtensor<[4],ui8>,
         !torch.vtensor<[4,3],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>,
         !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>) -> !torch.vtensor<[3],ui8>
  return %0 : !torch.vtensor<[3],ui8>
}

// -----

// b_zero_point must have the same shape as b_scale.
func.func @test_qlinearmatmul_b_zp_shape_mismatch(
    %arg0: !torch.vtensor<[2,4],ui8>, %arg1: !torch.vtensor<[],f32>,
    %arg2: !torch.vtensor<[],ui8>, %arg3: !torch.vtensor<[4,3],ui8>,
    %arg4: !torch.vtensor<[3],f32>, %arg5: !torch.vtensor<[1,3],ui8>,
    %arg6: !torch.vtensor<[],f32>, %arg7: !torch.vtensor<[],ui8>)
    -> !torch.vtensor<[2,3],ui8>
    attributes {torch.onnx_meta.opset_version = 21 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.QLinearMatMul"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7)
      : (!torch.vtensor<[2,4],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>,
         !torch.vtensor<[4,3],ui8>, !torch.vtensor<[3],f32>, !torch.vtensor<[1,3],ui8>,
         !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>) -> !torch.vtensor<[2,3],ui8>
  return %0 : !torch.vtensor<[2,3],ui8>
}

// -----

// b_zero_point must have the same dtype as b.
func.func @test_qlinearmatmul_b_zp_dtype_mismatch(
    %arg0: !torch.vtensor<[2,4],ui8>, %arg1: !torch.vtensor<[],f32>,
    %arg2: !torch.vtensor<[],ui8>, %arg3: !torch.vtensor<[4,3],si8>,
    %arg4: !torch.vtensor<[3],f32>, %arg5: !torch.vtensor<[3],ui8>,
    %arg6: !torch.vtensor<[],f32>, %arg7: !torch.vtensor<[],ui8>)
    -> !torch.vtensor<[2,3],ui8>
    attributes {torch.onnx_meta.opset_version = 21 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.QLinearMatMul"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7)
      : (!torch.vtensor<[2,4],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>,
         !torch.vtensor<[4,3],si8>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],ui8>,
         !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>) -> !torch.vtensor<[2,3],ui8>
  return %0 : !torch.vtensor<[2,3],ui8>
}

// -----

// Quantization parameters of `b` that vary along a dimension other than the
// last one (here: per-row) are not supported.
func.func @test_qlinearmatmul_b_per_row_params(
    %arg0: !torch.vtensor<[2,4],ui8>, %arg1: !torch.vtensor<[],f32>,
    %arg2: !torch.vtensor<[],ui8>, %arg3: !torch.vtensor<[4,3],ui8>,
    %arg4: !torch.vtensor<[4,1],f32>, %arg5: !torch.vtensor<[4,1],ui8>,
    %arg6: !torch.vtensor<[],f32>, %arg7: !torch.vtensor<[],ui8>)
    -> !torch.vtensor<[2,3],ui8>
    attributes {torch.onnx_meta.opset_version = 21 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.QLinearMatMul"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7)
      : (!torch.vtensor<[2,4],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>,
         !torch.vtensor<[4,3],ui8>, !torch.vtensor<[4,1],f32>, !torch.vtensor<[4,1],ui8>,
         !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>) -> !torch.vtensor<[2,3],ui8>
  return %0 : !torch.vtensor<[2,3],ui8>
}

// -----

// A 1-D a_scale must have length M (per-row quantization of `a`), here it has
// length K.
func.func @test_qlinearmatmul_a_1d_scale_wrong_length(
    %arg0: !torch.vtensor<[2,4],ui8>, %arg1: !torch.vtensor<[4],f32>,
    %arg2: !torch.vtensor<[4],ui8>, %arg3: !torch.vtensor<[4,3],ui8>,
    %arg4: !torch.vtensor<[],f32>, %arg5: !torch.vtensor<[],ui8>,
    %arg6: !torch.vtensor<[],f32>, %arg7: !torch.vtensor<[],ui8>)
    -> !torch.vtensor<[2,3],ui8>
    attributes {torch.onnx_meta.opset_version = 21 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.QLinearMatMul"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7)
      : (!torch.vtensor<[2,4],ui8>, !torch.vtensor<[4],f32>, !torch.vtensor<[4],ui8>,
         !torch.vtensor<[4,3],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>,
         !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>) -> !torch.vtensor<[2,3],ui8>
  return %0 : !torch.vtensor<[2,3],ui8>
}

// -----

// Per-element quantization parameters of `a` are not allowed by the ONNX spec.
func.func @test_qlinearmatmul_a_per_element_params(
    %arg0: !torch.vtensor<[2,4],ui8>, %arg1: !torch.vtensor<[2,4],f32>,
    %arg2: !torch.vtensor<[2,4],ui8>, %arg3: !torch.vtensor<[4,3],ui8>,
    %arg4: !torch.vtensor<[],f32>, %arg5: !torch.vtensor<[],ui8>,
    %arg6: !torch.vtensor<[],f32>, %arg7: !torch.vtensor<[],ui8>)
    -> !torch.vtensor<[2,3],ui8>
    attributes {torch.onnx_meta.opset_version = 21 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.QLinearMatMul"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7)
      : (!torch.vtensor<[2,4],ui8>, !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],ui8>,
         !torch.vtensor<[4,3],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>,
         !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>) -> !torch.vtensor<[2,3],ui8>
  return %0 : !torch.vtensor<[2,3],ui8>
}

// -----

// a_zero_point must have the same shape as a_scale.
func.func @test_qlinearmatmul_a_zp_shape_mismatch(
    %arg0: !torch.vtensor<[2,4],ui8>, %arg1: !torch.vtensor<[2],f32>,
    %arg2: !torch.vtensor<[2,1],ui8>, %arg3: !torch.vtensor<[4,3],ui8>,
    %arg4: !torch.vtensor<[],f32>, %arg5: !torch.vtensor<[],ui8>,
    %arg6: !torch.vtensor<[],f32>, %arg7: !torch.vtensor<[],ui8>)
    -> !torch.vtensor<[2,3],ui8>
    attributes {torch.onnx_meta.opset_version = 21 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.QLinearMatMul"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7)
      : (!torch.vtensor<[2,4],ui8>, !torch.vtensor<[2],f32>, !torch.vtensor<[2,1],ui8>,
         !torch.vtensor<[4,3],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>,
         !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>) -> !torch.vtensor<[2,3],ui8>
  return %0 : !torch.vtensor<[2,3],ui8>
}

// -----

// Per-element quantization parameters of `b` are not allowed by the ONNX spec.
func.func @test_qlinearmatmul_b_per_element_params(
    %arg0: !torch.vtensor<[2,4],ui8>, %arg1: !torch.vtensor<[],f32>,
    %arg2: !torch.vtensor<[],ui8>, %arg3: !torch.vtensor<[4,3],ui8>,
    %arg4: !torch.vtensor<[4,3],f32>, %arg5: !torch.vtensor<[4,3],ui8>,
    %arg6: !torch.vtensor<[],f32>, %arg7: !torch.vtensor<[],ui8>)
    -> !torch.vtensor<[2,3],ui8>
    attributes {torch.onnx_meta.opset_version = 21 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.QLinearMatMul"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7)
      : (!torch.vtensor<[2,4],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>,
         !torch.vtensor<[4,3],ui8>, !torch.vtensor<[4,3],f32>, !torch.vtensor<[4,3],ui8>,
         !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>) -> !torch.vtensor<[2,3],ui8>
  return %0 : !torch.vtensor<[2,3],ui8>
}

// -----

// The output `y` only supports per-tensor quantization.
func.func @test_qlinearmatmul_y_not_per_tensor(
    %arg0: !torch.vtensor<[2,4],ui8>, %arg1: !torch.vtensor<[],f32>,
    %arg2: !torch.vtensor<[],ui8>, %arg3: !torch.vtensor<[4,3],ui8>,
    %arg4: !torch.vtensor<[],f32>, %arg5: !torch.vtensor<[],ui8>,
    %arg6: !torch.vtensor<[3],f32>, %arg7: !torch.vtensor<[3],ui8>)
    -> !torch.vtensor<[2,3],ui8>
    attributes {torch.onnx_meta.opset_version = 21 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.QLinearMatMul"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7)
      : (!torch.vtensor<[2,4],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>,
         !torch.vtensor<[4,3],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>,
         !torch.vtensor<[3],f32>, !torch.vtensor<[3],ui8>) -> !torch.vtensor<[2,3],ui8>
  return %0 : !torch.vtensor<[2,3],ui8>
}

// -----

// float8 inputs (allowed by opset 21) are not supported, since the
// quantize/dequantize helpers only handle integer types. The pattern must
// fail before any IR is created.
func.func @test_qlinearmatmul_f8_input(
    %arg0: !torch.vtensor<[2,4],f8E4M3FN>, %arg1: !torch.vtensor<[],f32>,
    %arg2: !torch.vtensor<[],f8E4M3FN>, %arg3: !torch.vtensor<[4,3],ui8>,
    %arg4: !torch.vtensor<[],f32>, %arg5: !torch.vtensor<[],ui8>,
    %arg6: !torch.vtensor<[],f32>, %arg7: !torch.vtensor<[],ui8>)
    -> !torch.vtensor<[2,3],ui8>
    attributes {torch.onnx_meta.opset_version = 21 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.QLinearMatMul"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7)
      : (!torch.vtensor<[2,4],f8E4M3FN>, !torch.vtensor<[],f32>, !torch.vtensor<[],f8E4M3FN>,
         !torch.vtensor<[4,3],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>,
         !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>) -> !torch.vtensor<[2,3],ui8>
  return %0 : !torch.vtensor<[2,3],ui8>
}

// -----

// A float8 output is not supported either.
func.func @test_qlinearmatmul_f8_output(
    %arg0: !torch.vtensor<[2,4],ui8>, %arg1: !torch.vtensor<[],f32>,
    %arg2: !torch.vtensor<[],ui8>, %arg3: !torch.vtensor<[4,3],ui8>,
    %arg4: !torch.vtensor<[],f32>, %arg5: !torch.vtensor<[],ui8>,
    %arg6: !torch.vtensor<[],f32>, %arg7: !torch.vtensor<[],f8E4M3FN>)
    -> !torch.vtensor<[2,3],f8E4M3FN>
    attributes {torch.onnx_meta.opset_version = 21 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.QLinearMatMul"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7)
      : (!torch.vtensor<[2,4],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>,
         !torch.vtensor<[4,3],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>,
         !torch.vtensor<[],f32>, !torch.vtensor<[],f8E4M3FN>) -> !torch.vtensor<[2,3],f8E4M3FN>
  return %0 : !torch.vtensor<[2,3],f8E4M3FN>
}
