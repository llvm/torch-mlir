// RUN: npcomp-opt -torch-refine-types -split-input-file %s | FileCheck %s

// CHECK-LABEL:   func @f(
// CHECK-SAME:            %[[ARG:.*]]: tensor<2x3x?xf32>) -> tensor<*x!numpy.any_dtype> {
// CHECK:           %[[SHAPED:.*]] = numpy.tensor_static_info_cast %[[ARG]] : tensor<2x3x?xf32> to tensor<2x3x?xf32>
// CHECK:           %[[SHAPE_ERASED:.*]] = numpy.tensor_static_info_cast %[[SHAPED]] : tensor<2x3x?xf32> to tensor<*x!numpy.any_dtype>
// CHECK:           return %[[SHAPE_ERASED]] : tensor<*x!numpy.any_dtype>
func @f(%arg0: tensor<2x3x?xf32>) -> tensor<*x!numpy.any_dtype> {
  %0 = numpy.tensor_static_info_cast %arg0 : tensor<2x3x?xf32> to tensor<*x!numpy.any_dtype>
  return %0 : tensor<*x!numpy.any_dtype>
}

// -----

// CHECK-LABEL:   func @f(
// CHECK-SAME:            %[[ARG:.*]]: tensor<2x3x?xf32>) -> tensor<*x!numpy.any_dtype> {
// CHECK:           %[[SHAPED:.*]] = "aten.tanh"(%[[ARG]]) : (tensor<2x3x?xf32>) -> tensor<2x3x?xf32>
// CHECK:           %[[SHAPE_ERASED:.*]] = numpy.tensor_static_info_cast %[[SHAPED]] : tensor<2x3x?xf32> to tensor<*x!numpy.any_dtype>
// CHECK:           return %[[SHAPE_ERASED]] : tensor<*x!numpy.any_dtype>
func @f(%arg0: tensor<2x3x?xf32>) -> tensor<*x!numpy.any_dtype> {
  %1 = "aten.tanh"(%arg0) : (tensor<2x3x?xf32>) -> tensor<*x!numpy.any_dtype>
  return %1 : tensor<*x!numpy.any_dtype>
}

// -----

// CHECK-LABEL:   func @f
func @f(%arg0: tensor<2x3x?xf32>) -> tensor<*x!numpy.any_dtype> {
  // Check propagation through multiple ops.
  // CHECK:           "aten.tanh"(%{{.*}}) : (tensor<2x3x?xf32>) -> tensor<2x3x?xf32>
  // CHECK:           "aten.tanh"(%{{.*}}) : (tensor<2x3x?xf32>) -> tensor<2x3x?xf32>
  // CHECK:           "aten.tanh"(%{{.*}}) : (tensor<2x3x?xf32>) -> tensor<2x3x?xf32>
  %1 = "aten.tanh"(%arg0) : (tensor<2x3x?xf32>) -> tensor<*x!numpy.any_dtype>
  %2 = "aten.tanh"(%1) : (tensor<*x!numpy.any_dtype>) -> tensor<*x!numpy.any_dtype>
  %3 = "aten.tanh"(%2) : (tensor<*x!numpy.any_dtype>) -> tensor<*x!numpy.any_dtype>
  return %3 : tensor<*x!numpy.any_dtype>
}
