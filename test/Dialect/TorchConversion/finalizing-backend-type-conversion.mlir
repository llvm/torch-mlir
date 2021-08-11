// RUN: npcomp-opt %s -torch-finalizing-backend-type-conversion -split-input-file -verify-diagnostics -allow-unregistered-dialect | FileCheck %s

// This test is largely copied from `finalizing-bufferize` upstream, as it
// covers the same scope.

// CHECK-LABEL:   func @eliminate_materializations(
// CHECK-SAME:                                     %[[ARG:.*]]: tensor<f32>) -> tensor<f32> {
// CHECK:           return %[[ARG]] : tensor<f32>
func @eliminate_materializations(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = torch_c.from_builtin_tensor %arg0 : tensor<f32> -> !torch.vtensor<[],f32>
  %1 = torch_c.to_builtin_tensor %0 : !torch.vtensor<[],f32> -> tensor<f32>
  return %1 : tensor<f32>
}

// Do a basic check of other types. Under the hood they all take the same
// code paths as for !torch.vtensor, so we just spot-check them here.

// CHECK-LABEL:   func @eliminate_materializations$torch.bool(
// CHECK-SAME:                                     %[[ARG:.*]]: i1) -> i1 {
// CHECK:           return %[[ARG]] : i1
func @eliminate_materializations$torch.bool(%arg0: i1) -> i1 {
  %0 = torch_c.from_i1 %arg0
  %1 = torch_c.to_i1 %0
  return %1 : i1
}

// CHECK-LABEL:   func @eliminate_materializations$torch.int(
// CHECK-SAME:                                     %[[ARG:.*]]: i64) -> i64 {
// CHECK:           return %[[ARG]] : i64
func @eliminate_materializations$torch.int(%arg0: i64) -> i64 {
  %0 = torch_c.from_i64 %arg0
  %1 = torch_c.to_i64 %0
  return %1 : i64
}

// CHECK-LABEL:   func @eliminate_materializations$torch.float(
// CHECK-SAME:                                                 %[[ARG:.*]]: f64) -> f64 {
// CHECK:           return %[[ARG]] : f64
func @eliminate_materializations$torch.float(%arg0: f64) -> f64 {
  %0 = torch_c.from_f64 %arg0
  %1 = torch_c.to_f64 %0
  return %1 : f64
}

// CHECK-LABEL:   func @eliminate_materializations$torch.list(
// CHECK-SAME:                                                %[[ARG:.*]]: !iree.list<f64>) -> !iree.list<f64> {
// CHECK:           return %[[ARG]] : !iree.list<f64>
func @eliminate_materializations$torch.list(%arg0: !iree.list<f64>) -> !iree.list<f64> {
  %0 = torch_c.from_iree_list %arg0 : !iree.list<f64> -> !torch.list<!torch.float>
  %1 = torch_c.to_iree_list %0 : !torch.list<!torch.float> -> !iree.list<f64>
  return %1 : !iree.list<f64>
}

// -----

func @unable_to_convert_lone_buffer_cast() -> tensor<f32> {
  // expected-error @+1 {{failed to legalize operation 'test.source'}}
  %0 = "test.source"() : () -> !torch.vtensor<[],f32>
  %1 = torch_c.to_builtin_tensor %0 : !torch.vtensor<[],f32> -> tensor<f32>
  return %1 : tensor<f32>
}

// -----

func @unable_to_convert_lone_tensor_load(%arg0: tensor<f32>) {
  %0 = torch_c.from_builtin_tensor %arg0 : tensor<f32> -> !torch.vtensor<[],f32>
  // expected-error @+1 {{failed to legalize operation 'test.sink'}}
  "test.sink"(%0) : (!torch.vtensor<[],f32>) -> ()
  return
}
