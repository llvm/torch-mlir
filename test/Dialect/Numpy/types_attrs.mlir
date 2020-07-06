// RUN: npcomp-opt -split-input-file %s | npcomp-opt | FileCheck --dump-input=fail %s

// CHECK-LABEL: @ndarray_no_dtype
// CHECK: !numpy.ndarray<*:?>
func @ndarray_no_dtype(%arg0 : !numpy.ndarray<*:?>) -> !numpy.ndarray<*:?> {
  return %arg0 : !numpy.ndarray<*:?>
}

// -----
// CHECK-LABEL: @ndarray_dtype
// CHECK: !numpy.ndarray<*:i32>
func @ndarray_dtype(%arg0 : !numpy.ndarray<*:i32>) -> !numpy.ndarray<*:i32> {
  return %arg0 : !numpy.ndarray<*:i32>
}

// -----
// CHECK-LABEL: @ndarray_ranked
// CHECK: !numpy.ndarray<[1,?,3]:i32>
func @ndarray_ranked(%arg0 : !numpy.ndarray<[1,?,3]:i32>) -> !numpy.ndarray<[1,?,3]:i32> {
  return %arg0 : !numpy.ndarray<[1,?,3]:i32>
}
