// RUN: npcomp-run-mlir %s \
// RUN:   -invoke max \
// RUN:   -arg-value="dense<[1.0]> : tensor<1xf32>" \
// RUN:   -arg-value="dense<[3.0]> : tensor<1xf32>" \
// RUN:   -shared-libs=%npcomp_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s --check-prefix=MAX

// These ops share a lot of code paths. So we don't test the exact
// broadcasting behavior and error checking for all of them.

// MAX: output #0: dense<3.000000e+00> : tensor<1xf32>
func @max(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %0 =  tcf.max %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

