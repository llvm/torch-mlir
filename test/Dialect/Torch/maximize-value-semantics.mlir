// RUN: npcomp-opt -split-input-file %s -torch-maximize-value-semantics | FileCheck %s

// Basic case that can be resolved with local reasoning.
// This pass will eventually need to learn about aliasing relationships.
//
// This is taken from a test case from an e2e spike, and isn't intended to be
// particularly minimal or specifically test one thing, since the pass is
// currently just a handful of canonicalization patterns that are already
// tested elsewhere.

// CHECK-LABEL:   func @local(
// CHECK-SAME:                %[[ARG:.*]]: !torch.vtensor<[2,3,?],f32>) -> !torch.vtensor<[2,3,?],f32> {
// CHECK:           %[[RET:.*]] = torch.aten.tanh %[[ARG]] : !torch.vtensor<[2,3,?],f32> -> !torch.vtensor<[2,3,?],f32>
// CHECK:           return %[[RET]] : !torch.vtensor<[2,3,?],f32>
func @local(%arg0: !torch.vtensor<[2,3,?],f32>) -> !torch.vtensor<[2,3,?],f32> {
  %0 = torch.tensor_static_info_cast %arg0 : !torch.vtensor<[2,3,?],f32> to !torch.vtensor<[2,3,?],f32>
  %1 = torch.aten.tanh %0 : !torch.vtensor<[2,3,?],f32> -> !torch.vtensor<[2,3,?],f32>
  %2 = torch.copy.tensor %1 : !torch.vtensor<[2,3,?],f32> -> !torch.tensor<[2,3,?],f32>
  %3 = torch.tensor_static_info_cast %2 : !torch.tensor<[2,3,?],f32> to !torch.tensor
  %4 = torch.copy.tensor %2 : !torch.tensor<[2,3,?],f32> -> !torch.vtensor<[2,3,?],f32>
  return %4 : !torch.vtensor<[2,3,?],f32>
}
