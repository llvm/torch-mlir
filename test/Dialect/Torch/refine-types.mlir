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

// CHECK-LABEL:   func @f(
// CHECK-SAME:            %[[LHS:.*]]: tensor<2x?xf32>,
// CHECK-SAME:            %[[RHS:.*]]: tensor<?x?xf32>) -> tensor<*x!numpy.any_dtype> {
// CHECK:           %[[MM:.*]] = "aten.mm"(%[[LHS]], %[[RHS]]) : (tensor<2x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[SHAPE_ERASED:.*]] = numpy.tensor_static_info_cast %[[MM]] : tensor<?x?xf32> to tensor<*x!numpy.any_dtype>
// CHECK:           return %[[SHAPE_ERASED]] : tensor<*x!numpy.any_dtype>
func @f(%arg0: tensor<2x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<*x!numpy.any_dtype> {
  %1 = "aten.mm"(%arg0, %arg1) : (tensor<2x?xf32>, tensor<?x?xf32>) -> tensor<*x!numpy.any_dtype>
  return %1 : tensor<*x!numpy.any_dtype>
}

// -----

// CHECK-LABEL:   func @f(
// CHECK-SAME:            %[[INPUT:.*]]: tensor<?x3xf32>,
// CHECK-SAME:            %[[WEIGHT:.*]]: tensor<5x3xf32>,
// CHECK-SAME:            %[[BIAS:.*]]: tensor<5xf32>) -> tensor<*x!numpy.any_dtype> {
// CHECK:           %[[LINEAR:.*]] = "aten.linear"(%[[INPUT]], %[[WEIGHT]], %[[BIAS]]) : (tensor<?x3xf32>, tensor<5x3xf32>, tensor<5xf32>) -> tensor<?x?xf32>
// CHECK:           %[[SHAPE_ERASED:.*]] = numpy.tensor_static_info_cast %[[LINEAR]] : tensor<?x?xf32> to tensor<*x!numpy.any_dtype>
// CHECK:           return %[[SHAPE_ERASED]] : tensor<*x!numpy.any_dtype>
func @f(%arg0: tensor<?x3xf32>, %arg1: tensor<5x3xf32>, %arg2: tensor<5xf32>) -> tensor<*x!numpy.any_dtype> {
  %1 = "aten.linear"(%arg0, %arg1, %arg2) : (tensor<?x3xf32>, tensor<5x3xf32>, tensor<5xf32>) -> tensor<*x!numpy.any_dtype>
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

// -----

// CHECK-LABEL:   func @f
// CHECK: %[[ATEN:.*]] = "aten.tanh"(%{{.*}}) : (tensor<*x!numpy.any_dtype>) -> tensor<2x3x?xf32>
// CHECK: %[[CAST:.*]] = numpy.tensor_static_info_cast %[[ATEN]] : tensor<2x3x?xf32> to tensor<*x!numpy.any_dtype> 
// CHECK: return %[[CAST]] : tensor<*x!numpy.any_dtype>
func @f(%arg0: tensor<2x3x?xf32>) -> tensor<*x!numpy.any_dtype> {
  %cast = numpy.tensor_static_info_cast %arg0 : tensor<2x3x?xf32> to tensor<*x!numpy.any_dtype>
  br ^bb1(%cast: tensor<*x!numpy.any_dtype>)
^bb1(%arg1: tensor<*x!numpy.any_dtype>):
  %1 = "aten.tanh"(%arg1) : (tensor<*x!numpy.any_dtype>) -> tensor<*x!numpy.any_dtype>
  return %1 : tensor<*x!numpy.any_dtype>
}

// -----

// CHECK-LABEL:   func @f
// CHECK: func private @callee
// CHECK-NEXT: "aten.tanh"(%{{.*}}) : (tensor<*x!numpy.any_dtype>) -> tensor<2x3x?xf32>
func @f() {
  module {
    func private @callee(%arg0: tensor<*x!numpy.any_dtype>) {
      %1 = "aten.tanh"(%arg0) : (tensor<*x!numpy.any_dtype>) -> tensor<*x!numpy.any_dtype>
      return
    }
    func @caller(%arg0: tensor<2x3x?xf32>) {
      %cast = numpy.tensor_static_info_cast %arg0 : tensor<2x3x?xf32> to tensor<*x!numpy.any_dtype>
      call @callee(%cast) : (tensor<*x!numpy.any_dtype>) -> ()
      return
    }
  }
  return
}
