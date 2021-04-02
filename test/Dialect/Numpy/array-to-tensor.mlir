// RUN: npcomp-opt -split-input-file %s -numpy-array-to-tensor | FileCheck --dump-input=fail %s

// Basic case that can be resolved with local reasoning.
// This pass will eventually need to learn about aliasing relationships.
//
// This is taken from a test case from an e2e spike, and isn't intended to be
// particularly minimal or specifically test one thing, since the pass is
// currently just a handful of canonicalization patterns that are already
// tested elsewhere.

// CHECK-LABEL:   func @local(
// CHECK-SAME:                  %[[ARG:.*]]: tensor<2x3x?xf32>) -> tensor<*x!numpy.any_dtype> {
// CHECK:           %[[ERASED:.*]] = numpy.tensor_static_info_cast %[[ARG]] : tensor<2x3x?xf32> to tensor<*x!numpy.any_dtype>
// CHECK:           %[[RET:.*]] = "aten.tanh"(%[[ERASED]]) : (tensor<*x!numpy.any_dtype>) -> tensor<*x!numpy.any_dtype>
// CHECK:           return %[[RET]] : tensor<*x!numpy.any_dtype>
func @local(%arg0: tensor<2x3x?xf32>) -> tensor<*x!numpy.any_dtype> {
  %0 = numpy.create_array_from_tensor %arg0 : (tensor<2x3x?xf32>) -> !numpy.ndarray<[2,3,?]:f32>
  %1 = numpy.static_info_cast %0 : !numpy.ndarray<[2,3,?]:f32> to !numpy.ndarray<*:!numpy.any_dtype>
  %2 = numpy.copy_to_tensor %1 : (!numpy.ndarray<*:!numpy.any_dtype>) -> tensor<*x!numpy.any_dtype>
  %3 = "aten.tanh"(%2) : (tensor<*x!numpy.any_dtype>) -> tensor<*x!numpy.any_dtype>
  %4 = numpy.create_array_from_tensor %3 : (tensor<*x!numpy.any_dtype>) -> !numpy.ndarray<*:!numpy.any_dtype>
  %5 = numpy.copy_to_tensor %4 : (!numpy.ndarray<*:!numpy.any_dtype>) -> tensor<*x!numpy.any_dtype>
  return %5 : tensor<*x!numpy.any_dtype>
}
