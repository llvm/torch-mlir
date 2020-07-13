// RUN: npcomp-run-mlir %s \
// RUN:   -invoke constant_add \
// RUN:   -arg-value="dense<[3.0, 5.0]> : tensor<2xf32>" \
// RUN:   -shared-libs=%npcomp_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s

// CHECK: output #0: dense<[4.000000e+00, 7.000000e+00]> : tensor<2xf32>
func @constant_add(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = constant dense<[1.0, 2.0]> : tensor<2xf32>
  %1 = "tcf.add"(%arg0, %0) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %1 : tensor<2xf32>
}