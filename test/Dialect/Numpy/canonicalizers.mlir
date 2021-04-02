// RUN: npcomp-opt -split-input-file %s -canonicalize | FileCheck --dump-input=fail %s

// CHECK-LABEL: func @elideCreateRedundantArrayFromTensor
func @elideCreateRedundantArrayFromTensor() -> tensor<2xf64> {
  // CHECK: %[[CST:.*]] = constant
  // CHECK-NOT: numpy.create_array_from_tensor
  // CHECK-NOT: numpy.copy_to_tensor
  %cst = constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
  %0 = numpy.create_array_from_tensor %cst : (tensor<2xf64>) -> !numpy.ndarray<[2]:f64>
  %1 = numpy.copy_to_tensor %0 : (!numpy.ndarray<[2]:f64>) -> tensor<2xf64>
  // CHECK: return %[[CST]]
  return %1 : tensor<2xf64>
}

// This test verifies that the very trivial elision is not overly aggressive.
// Note that in this example, it is still safe to remove the copy, but the
// analysis has not yet been written to do that safely.
// CHECK-LABEL: func @elideCreateRedundantArrayFromTensorNonTrivial
func @elideCreateRedundantArrayFromTensorNonTrivial() -> (tensor<2xf64>, tensor<2xf64>) {
  // CHECK: numpy.create_array_from_tensor
  // CHECK: numpy.copy_to_tensor
  %cst = constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
  %0 = numpy.create_array_from_tensor %cst : (tensor<2xf64>) -> !numpy.ndarray<[2]:f64>
  %1 = numpy.copy_to_tensor %0 : (!numpy.ndarray<[2]:f64>) -> tensor<2xf64>
  %2 = numpy.copy_to_tensor %0 : (!numpy.ndarray<[2]:f64>) -> tensor<2xf64>
  return %1, %2 : tensor<2xf64>, tensor<2xf64>
}

// CHECK-LABEL:   func @commuteStaticInfoCastOpWithCreateArrayFromTensorOp(
// CHECK-SAME:                                                             %[[TENSOR:.*]]: tensor<2x3x?xf32>) -> !numpy.ndarray<*:!numpy.any_dtype> {
// CHECK:           %[[ERASED_TENSOR:.*]] = numpy.tensor_static_info_cast %[[TENSOR]] : tensor<2x3x?xf32> to tensor<*x!numpy.any_dtype>
// CHECK:           %[[ERASED_ARRAY:.*]] = numpy.create_array_from_tensor %[[ERASED_TENSOR]] : (tensor<*x!numpy.any_dtype>) -> !numpy.ndarray<*:!numpy.any_dtype>
// CHECK:           return %[[ERASED_ARRAY]] : !numpy.ndarray<*:!numpy.any_dtype>
func @commuteStaticInfoCastOpWithCreateArrayFromTensorOp(%arg0: tensor<2x3x?xf32>) -> !numpy.ndarray<*:!numpy.any_dtype> {
  %0 = numpy.create_array_from_tensor %arg0 : (tensor<2x3x?xf32>) -> !numpy.ndarray<[2,3,?]:f32>
  %1 = numpy.static_info_cast %0 : !numpy.ndarray<[2,3,?]:f32> to !numpy.ndarray<*:!numpy.any_dtype>
  return %1 : !numpy.ndarray<*:!numpy.any_dtype>
}
