// XFAIL: *
// RUN: npcomp-run-mlir -input %s -invoke basic -arg-value="dense<[1.0]> : tensor<1xf32>" \
// RUN:   -shared-libs=%npcomp_runtime_shlib 2>&1 | FileCheck %s

// CHECK: SUCCESS
func @basic(%arg0: tensor<?xf32>) {
  %0 = "tcf.add"(%arg0, %arg0) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return
}

