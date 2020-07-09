// RUN: npcomp-run-mlir -input %s \
// RUN:   -invoke multi_output \
// RUN:   -arg-value="dense<1.0> : tensor<f32>" \
// RUN:   -shared-libs=%npcomp_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s

// CHECK: output #0: dense<2.000000e+00> : tensor<f32>
// CHECK: output #1: dense<2.000000e+00> : tensor<f32>
func @multi_output(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  %0 = "tcf.add"(%arg0, %arg0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0, %0 : tensor<f32>, tensor<f32>
}
