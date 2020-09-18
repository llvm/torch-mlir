// RUN: npcomp-run-mlir %s \
// RUN:   -invoke mixed_rank \
// RUN:   -arg-value="dense<[1.0]> : tensor<1xf32>" \
// RUN:   -arg-value="dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>" \
// RUN:   -shared-libs=%npcomp_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s

// CHECK: output #0: dense<[
// CHECK-SAME: [2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00]]> : tensor<2x2xf32>
func @mixed_rank(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcf.add %arg0, %arg1 : (tensor<?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}