// RUN: refback-run %s \
// RUN:   -invoke constant \
// RUN:   -shared-libs=%npcomp_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s

// CHECK: output #0: dense<1.000000e+00> : tensor<f32>
func @constant() -> tensor<f32> {
  %0 = constant dense<1.0> : tensor<f32>
  return %0 : tensor<f32>
}
