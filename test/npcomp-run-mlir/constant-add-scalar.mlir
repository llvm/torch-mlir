// RUN: npcomp-run-mlir %s \
// RUN:   -invoke constant_add_scalar \
// RUN:   -arg-value="dense<3.0> : tensor<f32>" \
// RUN:   -shared-libs=%npcomp_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s

// CHECK: output #0: dense<4.000000e+00> : tensor<f32>
func @constant_add_scalar(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = constant dense<1.0> : tensor<f32>
  %1 = tcf.add %arg0, %0 : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %1 : tensor<f32>
}