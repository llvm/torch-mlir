// RUN: npcomp-run-mlir %s \
// RUN:   -invoke broadcast \
// RUN:   -arg-value="dense<[[1.0], [10.0]]> : tensor<2x1xf32>" \
// RUN:   -arg-value="dense<[[3.0, 4.0]]> : tensor<1x2xf32>" \
// RUN:   -shared-libs=%npcomp_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s

//   2x1     1x2       2x2
//  [ 1] + [3, 4] == [ 4,  5]
//  [10]          == [13, 14]

// CHECK: output #0: dense<[
// CHECK-SAME: [4.000000e+00, 5.000000e+00], [1.300000e+01, 1.400000e+01]]> : tensor<2x2xf32>
func @broadcast(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcf.add %arg0, %arg1 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}