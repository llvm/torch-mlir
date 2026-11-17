// RUN: torch-mlir-opt %s -split-input-file -verify-diagnostics -convert-torch-onnx-to-torch

func.func @test_quantizelinear_si32_input(%arg0: !torch.vtensor<[6],si32>, %arg1: !torch.vtensor<[],f32>, %arg2: !torch.vtensor<[],si8>) -> !torch.vtensor<[6],si8> attributes {torch.onnx_meta.opset_version = 13 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.QuantizeLinear"(%arg0, %arg1, %arg2) : (!torch.vtensor<[6],si32>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[6],si8>
  return %0 : !torch.vtensor<[6],si8>
}

// -----

func.func @test_quantizelinear_si32_scale(%arg0: !torch.vtensor<[4,3],f32>, %arg1: !torch.vtensor<[3],si32>, %arg2: !torch.vtensor<[3],si8>) -> !torch.vtensor<[4,3],si8> attributes {torch.onnx_meta.opset_version = 23 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.QuantizeLinear"(%arg0, %arg1, %arg2) : (!torch.vtensor<[4,3],f32>, !torch.vtensor<[3],si32>, !torch.vtensor<[3],si8>) -> !torch.vtensor<[4,3],si8>
  return %0 : !torch.vtensor<[4,3],si8>
}
