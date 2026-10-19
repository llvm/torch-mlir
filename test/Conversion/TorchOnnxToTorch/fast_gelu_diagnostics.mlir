// RUN: torch-mlir-opt %s -split-input-file -verify-diagnostics -convert-torch-onnx-to-torch

func.func @fast_gelu_missing_input() -> !torch.vtensor<[4],f32>
    attributes {torch.onnx_meta.opset_version = 17 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.FastGelu"() : () -> !torch.vtensor<[4],f32>
  return %0 : !torch.vtensor<[4],f32>
}

// -----

func.func @fast_gelu_extra_input(%input: !torch.vtensor<[4],f32>, %bias: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32>
    attributes {torch.onnx_meta.opset_version = 17 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.FastGelu"(%input, %bias, %bias) : (!torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32>
  return %0 : !torch.vtensor<[4],f32>
}

// -----

func.func @fast_gelu_non_tensor_input(%input: !torch.float) -> !torch.vtensor<[4],f32>
    attributes {torch.onnx_meta.opset_version = 17 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.FastGelu"(%input) : (!torch.float) -> !torch.vtensor<[4],f32>
  return %0 : !torch.vtensor<[4],f32>
}

// -----

func.func @fast_gelu_non_tensor_bias(%input: !torch.vtensor<[4],f32>, %bias: !torch.float) -> !torch.vtensor<[4],f32>
    attributes {torch.onnx_meta.opset_version = 17 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.FastGelu"(%input, %bias) : (!torch.vtensor<[4],f32>, !torch.float) -> !torch.vtensor<[4],f32>
  return %0 : !torch.vtensor<[4],f32>
}

// -----

func.func @fast_gelu_integer_result(%input: !torch.vtensor<[4],si32>) -> !torch.vtensor<[4],si32>
    attributes {torch.onnx_meta.opset_version = 17 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.FastGelu"(%input) : (!torch.vtensor<[4],si32>) -> !torch.vtensor<[4],si32>
  return %0 : !torch.vtensor<[4],si32>
}
