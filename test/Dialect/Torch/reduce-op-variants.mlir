// RUN: npcomp-opt -torch-reduce-op-variants  %s | FileCheck %s

// CHECK-LABEL:   func @convert_to_value_semantic_tensors(
// CHECK-SAME:                                       %[[ARG:.*]]: !torch.tensor<[],f32>) -> !torch.tensor<[],f32> {
// CHECK:           %[[OPERAND_TENSOR:.*]] = torch.copy.tensor %[[ARG]] : !torch.tensor<[],f32> -> !torch.vtensor<[],f32>
// CHECK:           %[[RESULT_TENSOR:.*]] = torch.aten.tanh %[[OPERAND_TENSOR]] : !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
// CHECK:           %[[RET:.*]] = torch.copy.tensor %[[RESULT_TENSOR]] : !torch.vtensor<[],f32> -> !torch.tensor<[],f32>
// CHECK:           return %[[RET]] : !torch.tensor<[],f32>
func @convert_to_value_semantic_tensors(%arg0: !torch.tensor<[],f32>) -> !torch.tensor<[],f32> {
  %0 = torch.aten.tanh %arg0 : !torch.tensor<[],f32> -> !torch.tensor<[],f32>
  return %0 : !torch.tensor<[],f32>
}


// CHECK-LABEL:   func @reduce_trailing_underscore_inplace_variant(
// CHECK-SAME:                          %[[ARG0:.*]]: !torch.tensor<[2,2],f32>,
// CHECK-SAME:                          %[[ARG1:.*]]: !torch.tensor<[2,2],f32>) -> (!torch.tensor<[2,2],f32>, !torch.tensor<[2,2],f32>) {
// CHECK:           %[[C1:.*]] = torch.constant.int 1
// CHECK:           %[[TENSOR0:.*]] = torch.copy.tensor %[[ARG0]] : !torch.tensor<[2,2],f32> -> !torch.vtensor<[2,2],f32>
// CHECK:           %[[TENSOR1:.*]] = torch.copy.tensor %[[ARG1]] : !torch.tensor<[2,2],f32> -> !torch.vtensor<[2,2],f32>
// CHECK:           %[[TENSOR_RESULT:.*]] = torch.aten.add.Tensor %[[TENSOR0]], %[[TENSOR1]], %[[C1]] : !torch.vtensor<[2,2],f32>, !torch.vtensor<[2,2],f32>, !torch.int -> !torch.vtensor<[2,2],f32>
// Note: This somewhat redundant conversion back and forth
// (which is cleaned up by canonicalization) is an artifact of two patterns
// being applied in sequence.
// CHECK:           %[[ARRAY_RESULT:.*]] = torch.copy.tensor %[[TENSOR_RESULT]] : !torch.vtensor<[2,2],f32> -> !torch.tensor<[2,2],f32>
// CHECK:           %[[TENSOR_AGAIN:.*]] = torch.copy.tensor %[[ARRAY_RESULT]] : !torch.tensor<[2,2],f32> -> !torch.vtensor<[2,2],f32>
// CHECK:           torch.overwrite.tensor %[[TENSOR_AGAIN]] overwrites %[[ARG0]] : !torch.vtensor<[2,2],f32>, !torch.tensor<[2,2],f32>
// CHECK:           return %[[ARG0]], %[[ARG0]] : !torch.tensor<[2,2],f32>, !torch.tensor<[2,2],f32>
func @reduce_trailing_underscore_inplace_variant(%arg0: !torch.tensor<[2,2],f32>, %arg1: !torch.tensor<[2,2],f32>) -> (!torch.tensor<[2,2],f32>, !torch.tensor<[2,2],f32>) {
  %c1 = torch.constant.int 1
  %0 = torch.aten.add_.Tensor %arg0, %arg1, %c1 : !torch.tensor<[2,2],f32>, !torch.tensor<[2,2],f32>, !torch.int -> !torch.tensor<[2,2],f32>
  return %0, %arg0 : !torch.tensor<[2,2],f32>, !torch.tensor<[2,2],f32>
}

// CHECK-LABEL:   func @torch.tensor.literal() -> !torch.tensor {
// CHECK:           %[[VTENSOR:.*]] = torch.vtensor.literal(dense<0.000000e+00> : tensor<7xf32>) : !torch.vtensor<[7],f32>
// CHECK:           %[[SIZES_ERASED:.*]] = torch.tensor_static_info_cast %[[VTENSOR]] : !torch.vtensor<[7],f32> to !torch.vtensor
// CHECK:           %[[TENSOR:.*]] = torch.copy.tensor %[[SIZES_ERASED]] : !torch.vtensor -> !torch.tensor
// CHECK:           return %[[TENSOR]] : !torch.tensor
func @torch.tensor.literal() -> !torch.tensor {
  %0 = torch.tensor.literal(dense<0.0> : tensor<7xf32>) : !torch.tensor
  return %0 : !torch.tensor
}
