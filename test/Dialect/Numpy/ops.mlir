// RUN: npcomp-opt -split-input-file %s | npcomp-opt | FileCheck --dump-input=fail %s

// -----
// CHECK-LABEL: @builtin_ufunc
func @builtin_ufunc(%arg0 : tensor<3xf64>, %arg1 : tensor<3xf64>) -> tensor<3xf64> {
  %0 = numpy.builtin_ufunc_call<"numpy.add"> (%arg0, %arg1) : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
  return %0 : tensor<3xf64>
}

// CHECK-LABEL: @ndarray_tensor_bridging
func @ndarray_tensor_bridging(%arg0: !numpy.ndarray<[2,3]:f32>, %arg1: !numpy.ndarray<[2,3]:f32>, %arg2: tensor<2x3xf32>) {
  // CHECK-NEXT: numpy.copy_to_tensor
  %t = numpy.copy_to_tensor %arg1 : (!numpy.ndarray<[2,3]:f32>) -> tensor<2x3xf32>
  // CHECK-NEXT: numpy.create_array_from_tensor
  %a = numpy.create_array_from_tensor %arg2 : (tensor<2x3xf32>) -> !numpy.ndarray<[2,3]:f32>
  // CHECK-NEXT: numpy.overwrite_array
  numpy.overwrite_array %arg2 overwrites %arg0 : tensor<2x3xf32>, !numpy.ndarray<[2,3]:f32>
  return
}
