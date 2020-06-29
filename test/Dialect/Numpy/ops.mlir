// RUN: npcomp-opt -split-input-file %s | npcomp-opt | FileCheck --dump-input=fail %s

// -----
// CHECK-LABEL: @builtin_ufunc
func @builtin_ufunc(%arg0 : tensor<3xf64>, %arg1 : tensor<3xf64>) -> tensor<3xf64> {
  %0 = numpy.builtin_ufunc_call<"numpy.add"> (%arg0, %arg1) : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
  return %0 : tensor<3xf64>
}
