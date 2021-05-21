// RUN: npcomp-opt %s -torch-finalizing-builtin-tensorize -split-input-file -verify-diagnostics -allow-unregistered-dialect | FileCheck %s

// This test is largely copied from `finalizing-bufferize` upstream, as it
// covers the same scope.

// CHECK-LABEL:   func @eliminate_materializations(
// CHECK-SAME:                                     %[[ARG:.*]]: tensor<f32>) -> tensor<f32> {
// CHECK:           return %[[ARG]] : tensor<f32>
func @eliminate_materializations(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = torch.from_builtin_tensor %arg0 : tensor<f32> -> !torch.vtensor<[],f32>
  %1 = torch.to_builtin_tensor %0 : !torch.vtensor<[],f32> -> tensor<f32>
  return %1 : tensor<f32>
}

// -----

func @unable_to_convert_lone_buffer_cast() -> tensor<f32> {
  // expected-error @+1 {{failed to legalize operation 'test.source'}}
  %0 = "test.source"() : () -> !torch.vtensor<[],f32>
  %1 = torch.to_builtin_tensor %0 : !torch.vtensor<[],f32> -> tensor<f32>
  return %1 : tensor<f32>
}

// -----

func @unable_to_convert_lone_tensor_load(%arg0: tensor<f32>) {
  %0 = torch.from_builtin_tensor %arg0 : tensor<f32> -> !torch.vtensor<[],f32>
  // expected-error @+1 {{failed to legalize operation 'test.sink'}}
  "test.sink"(%0) : (!torch.vtensor<[],f32>) -> ()
  return
}
