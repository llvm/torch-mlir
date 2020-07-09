// RUN: npcomp-run-mlir -input %s \
// RUN:   -invoke basic \
// RUN:   -arg-value="dense<[1.0]> : tensor<1xf32>" \
// RUN:   -shared-libs=%npcomp_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s

// CHECK: output #0: dense<2.000000e+00> : tensor<1xf32>
func @basic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "tcf.add"(%arg0, %arg0) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

