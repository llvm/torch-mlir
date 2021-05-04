// RUN: npcomp-opt -torch-reduce-op-variants  %s | FileCheck %s

// CHECK-LABEL:   func @convert_to_immutable_tensors(
// CHECK-SAME:                                       %[[ARG:.*]]: !numpy.ndarray<[]:f32>) -> !numpy.ndarray<[]:f32> {
// CHECK:           %[[OPERAND_TENSOR:.*]] = numpy.copy_to_tensor %[[ARG]] : (!numpy.ndarray<[]:f32>) -> tensor<f32>
// CHECK:           %[[RESULT_TENSOR:.*]] = torch.aten.tanh %[[OPERAND_TENSOR]] : tensor<f32> -> tensor<f32>
// CHECK:           %[[RET:.*]] = numpy.create_array_from_tensor %[[RESULT_TENSOR]] : (tensor<f32>) -> !numpy.ndarray<[]:f32>
// CHECK:           return %[[RET]] : !numpy.ndarray<[]:f32>
func @convert_to_immutable_tensors(%arg0: !numpy.ndarray<[]:f32>) -> !numpy.ndarray<[]:f32> {
  %0 = torch.aten.tanh %arg0 : !numpy.ndarray<[]:f32> -> !numpy.ndarray<[]:f32>
  return %0 : !numpy.ndarray<[]:f32>
}


// CHECK-LABEL:   func @reduce_trailing_underscore_inplace_variant(
// CHECK-SAME:                          %[[ARG0:.*]]: !numpy.ndarray<[2,2]:f32>,
// CHECK-SAME:                          %[[ARG1:.*]]: !numpy.ndarray<[2,2]:f32>) -> (!numpy.ndarray<[2,2]:f32>, !numpy.ndarray<[2,2]:f32>) {
// CHECK:           %[[VAL_2:.*]] = constant 1 : i64
// CHECK:           %[[TENSOR0:.*]] = numpy.copy_to_tensor %[[ARG0]] : (!numpy.ndarray<[2,2]:f32>) -> tensor<2x2xf32>
// CHECK:           %[[TENSOR1:.*]] = numpy.copy_to_tensor %[[ARG1]] : (!numpy.ndarray<[2,2]:f32>) -> tensor<2x2xf32>
// CHECK:           %[[TENSOR_RESULT:.*]] = torch.aten.add.Tensor %[[TENSOR0]], %[[TENSOR1]], %[[VAL_2]] : tensor<2x2xf32>, tensor<2x2xf32>, i64 -> tensor<2x2xf32>
// Note: This somewhat redundant tensor->array->tensor conversion
// (which is cleaned up by canonicalization) is an artifact of two patterns
// being applied in sequence.
// CHECK:           %[[ARRAY_RESULT:.*]] = numpy.create_array_from_tensor %[[TENSOR_RESULT]] : (tensor<2x2xf32>) -> !numpy.ndarray<[2,2]:f32>
// CHECK:           %[[TENSOR_AGAIN:.*]] = numpy.copy_to_tensor %[[ARRAY_RESULT]] : (!numpy.ndarray<[2,2]:f32>) -> tensor<2x2xf32>
// CHECK:           numpy.overwrite_array %[[TENSOR_AGAIN]] overwrites %[[ARG0]] : tensor<2x2xf32>, !numpy.ndarray<[2,2]:f32>
// CHECK:           return %[[ARG0]], %[[ARG0]] : !numpy.ndarray<[2,2]:f32>, !numpy.ndarray<[2,2]:f32>
func @reduce_trailing_underscore_inplace_variant(%arg0: !numpy.ndarray<[2,2]:f32>, %arg1: !numpy.ndarray<[2,2]:f32>) -> (!numpy.ndarray<[2,2]:f32>, !numpy.ndarray<[2,2]:f32>) {
  %c1 = constant 1 : i64
  %0 = torch.aten.add_.Tensor %arg0, %arg1, %c1 : !numpy.ndarray<[2,2]:f32>, !numpy.ndarray<[2,2]:f32>, i64 -> !numpy.ndarray<[2,2]:f32>
  return %0, %arg0 : !numpy.ndarray<[2,2]:f32>, !numpy.ndarray<[2,2]:f32>
}
