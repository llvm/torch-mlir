// RUN: npcomp-run-mlir %s \
// RUN:   -invoke multi_output \
// RUN:   -arg-value="dense<1.0> : tensor<1xf32>" \
// RUN:   -shared-libs=%npcomp_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s

// CHECK: output #0: dense<2.000000e+00> : tensor<1xf32>
// CHECK: output #1: dense<2.000000e+00> : tensor<1xf32>
func @multi_output(%arg0: tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>) {
  %0 = tcf.add %arg0, %arg0 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %0, %0 : tensor<?xf32>, tensor<?xf32>
}
