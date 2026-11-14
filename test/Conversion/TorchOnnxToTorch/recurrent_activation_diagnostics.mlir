// RUN: torch-mlir-opt %s -split-input-file -verify-diagnostics -convert-torch-onnx-to-torch

// An unsupported activation name is a match failure, not a crash.
func.func @test_rnn_unknown_activation(%arg0: !torch.vtensor<[3,2,4],f32>, %arg1: !torch.vtensor<[1,3,4],f32>, %arg2: !torch.vtensor<[1,3,3],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:2 = torch.operator "onnx.RNN"(%arg0, %arg1, %arg2) {torch.onnx.hidden_size = 3 : si64, torch.onnx.activations = ["NotAnActivation"]} : (!torch.vtensor<[3,2,4],f32>, !torch.vtensor<[1,3,4],f32>, !torch.vtensor<[1,3,3],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>)
  return %0#0, %0#1 : !torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>
}

// -----

// activation_alpha must be a list of f32.
func.func @test_rnn_integer_activation_alpha(%arg0: !torch.vtensor<[3,2,4],f32>, %arg1: !torch.vtensor<[1,3,4],f32>, %arg2: !torch.vtensor<[1,3,3],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:2 = torch.operator "onnx.RNN"(%arg0, %arg1, %arg2) {torch.onnx.hidden_size = 3 : si64, torch.onnx.activations = ["LeakyRelu"], torch.onnx.activation_alpha = [1 : si64]} : (!torch.vtensor<[3,2,4],f32>, !torch.vtensor<[1,3,4],f32>, !torch.vtensor<[1,3,3],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>)
  return %0#0, %0#1 : !torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>
}
