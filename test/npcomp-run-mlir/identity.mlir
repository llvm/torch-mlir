// RUN: npcomp-run-mlir -input %s \
// RUN:   -invoke identity \
// RUN:   -arg-value="dense<1.0> : tensor<f32>" \
// RUN:   -shared-libs=%npcomp_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s

// CHECK: output #0: dense<1.000000e+00> : tensor<f32>
func @identity(%arg0: tensor<f32>) -> tensor<f32> {
  return %arg0 : tensor<f32>
}
