// RUN: npcomp-opt -split-input-file %s -verify-diagnostics -allow-unregistered-dialect -numpy-refine-public-return | FileCheck %s

// CHECK-LABEL:   func @basic(
// CHECK-SAME:              %[[ARG:.*]]: tensor<?xf32>) -> (tensor<?xf32>, i1, tensor<?xf32>) {
// CHECK:           %[[CTRUE:.*]] = constant true
// CHECK:           %[[CAST:.*]] = numpy.tensor_static_info_cast %[[ARG]] : tensor<?xf32> to tensor<*x!numpy.any_dtype>
// CHECK:           return %[[ARG]], %[[CTRUE]], %[[ARG]] : tensor<?xf32>, i1, tensor<?xf32>
func @basic(%arg0: tensor<?xf32>) -> (tensor<?xf32>, i1, tensor<*x!numpy.any_dtype>) {
  %ctrue = std.constant true
  %cast = numpy.tensor_static_info_cast %arg0 : tensor<?xf32> to tensor<*x!numpy.any_dtype>
  return %arg0, %ctrue, %cast : tensor<?xf32>, i1, tensor<*x!numpy.any_dtype>
}

// No conversion on private function.
// CHECK-LABEL:   func private @basic_private(
// CHECK-SAME:                                %[[ARG:.*]]: tensor<?xf32>) -> (tensor<?xf32>, i1, tensor<*x!numpy.any_dtype>) {
// CHECK:           %[[CTRUE:.*]] = constant true
// CHECK:           %[[CASTED:.*]] = numpy.tensor_static_info_cast %[[ARG]] : tensor<?xf32> to tensor<*x!numpy.any_dtype>
// CHECK:           return %[[ARG]], %[[CTRUE]], %[[CASTED]] : tensor<?xf32>, i1, tensor<*x!numpy.any_dtype>
func private @basic_private(%arg0: tensor<?xf32>) -> (tensor<?xf32>, i1, tensor<*x!numpy.any_dtype>) {
  %ctrue = std.constant true
  %cast = numpy.tensor_static_info_cast %arg0 : tensor<?xf32> to tensor<*x!numpy.any_dtype>
  return %arg0, %ctrue, %cast : tensor<?xf32>, i1, tensor<*x!numpy.any_dtype>
}


// -----

// Call to public function.
// expected-error @+1 {{unimplemented}}
func @called(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  return %arg0 : tensor<*xf32>
}

func private @caller(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = call @called(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

// Multiple returns.
// expected-error @+1 {{unimplemented}}
func @called(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %ctrue = constant true
  cond_br %ctrue, ^bb1, ^bb2
^bb1:
  return %arg0 : tensor<*xf32>
^bb2:
  return %arg0 : tensor<*xf32>
}
