// RUN: npcomp-opt -split-input-file %s | npcomp-opt | FileCheck --dump-input=fail %s
// -----
// CHECK-LABEL: @builtin_ufunc
module @builtin_ufunc {
  // CHECK: numpy.builtin_ufunc @numpy.add
  numpy.builtin_ufunc @numpy.add
  // CHECK: numpy.builtin_ufunc @numpy.custom_sub {some_attr = "foobar"}
  numpy.builtin_ufunc @numpy.custom_sub { some_attr = "foobar" }
}

// -----
// CHECK-LABEL: @example_generic_ufunc
module @example_generic_ufunc {
  // CHECK: numpy.generic_ufunc @numpy.add(
  numpy.generic_ufunc @numpy.add (
    // CHECK-SAME: overload(%arg0: i32, %arg1: i32) -> i32 {
    overload(%arg0: i32, %arg1: i32) -> i32 {
      // CHECK: addi
      %0 = addi %arg0, %arg1 : i32
      numpy.ufunc_return %0 : i32
    },
    // CHECK: overload(%arg0: f32, %arg1: f32) -> f32 {
    overload(%arg0: f32, %arg1: f32) -> f32 {
      // CHECK: addf
      %0 = addf %arg0, %arg1 : f32
      numpy.ufunc_return %0 : f32
    }
  )
}
