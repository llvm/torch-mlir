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

// CHECK-LABEL: @static_info_cast
func @static_info_cast(%arg0: !numpy.ndarray<[2,3]:f32>, %arg1: !numpy.ndarray<[?,3]:f32>, %arg2: !numpy.ndarray<*:f32>) {
  // CHECK-NEXT: numpy.static_info_cast %arg0 : !numpy.ndarray<[2,3]:f32> to !numpy.ndarray<*:!numpy.any_dtype>
  %0 = numpy.static_info_cast %arg0 : !numpy.ndarray<[2,3]:f32> to !numpy.ndarray<*:!numpy.any_dtype>
  // CHECK-NEXT: numpy.static_info_cast %arg1 : !numpy.ndarray<[?,3]:f32> to !numpy.ndarray<[7,3]:f32>
  %1 = numpy.static_info_cast %arg1 : !numpy.ndarray<[?,3]:f32> to !numpy.ndarray<[7,3]:f32>
  // CHECK-NEXT: numpy.static_info_cast %arg2 : !numpy.ndarray<*:f32> to !numpy.ndarray<[?,?]:f32>
  %2 = numpy.static_info_cast %arg2 : !numpy.ndarray<*:f32> to !numpy.ndarray<[?,?]:f32>
  return
}

// CHECK-LABEL: @tensor_static_info_cast
func @tensor_static_info_cast(%arg0: tensor<2x3xf32>, %arg1: tensor<?x3xf32>, %arg2: tensor<*xf32>) {
  // CHECK-NEXT: numpy.tensor_static_info_cast %arg0 : tensor<2x3xf32> to tensor<*x!numpy.any_dtype>
  %0 = numpy.tensor_static_info_cast %arg0 : tensor<2x3xf32> to tensor<*x!numpy.any_dtype>
  // CHECK-NEXT: numpy.tensor_static_info_cast %arg1 : tensor<?x3xf32> to tensor<7x3xf32>
  %1 = numpy.tensor_static_info_cast %arg1 : tensor<?x3xf32> to tensor<7x3xf32>
  // CHECK-NEXT: numpy.tensor_static_info_cast %arg2 : tensor<*xf32> to tensor<?x?xf32>
  %2 = numpy.tensor_static_info_cast %arg2 : tensor<*xf32> to tensor<?x?xf32>
  return
}
