// RUN: npcomp-run-mlir %s \
// RUN:   -invoke conv_2d_nchw_bias \
// RUN:   -arg-value="dense<[[[[1.2]]]]> : tensor<1x1x1x1xf32>" \
// RUN:   -arg-value="dense<[[[[3.4]]]]> : tensor<1x1x1x1xf32>" \
// RUN:   -arg-value="dense<[[[[5.6]]]]> : tensor<1x1x1x1xf32>" \
// RUN:   -shared-libs=%npcomp_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s

// RUN: npcomp-run-mlir %s \
// RUN:   -invoke conv_2d_nchw_bias \
// RUN:   -arg-value="dense<[[[[1.2, 3.4], [5.6, 7.8]]]]> : tensor<1x1x2x2xf32>" \
// RUN:   -arg-value="dense<[[[[3.4]]]]> : tensor<1x1x1x1xf32>" \
// RUN:   -arg-value="dense<[[[[4.0, 3.0], [2.0, 1.0]]]]> : tensor<1x1x2x2xf32>" \
// RUN:   -shared-libs=%npcomp_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s --check-prefix=TINY_SQUARE

// RUN: npcomp-run-mlir %s \
// RUN:   -invoke conv_2d_nchw_bias \
// RUN:   -arg-value="dense<0.0> : tensor<1x1x1x1xf32>" \
// RUN:   -arg-value="dense<0.0> : tensor<1x1x1x1xf32>" \
// RUN:   -arg-value="dense<0.0> : tensor<1x1x1x1xf32>" \
// RUN:   -shared-libs=%npcomp_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ZEROS

// Basic correctness checks:
// [1.2 3.4] * [5.6] + [4.0 3.0] = [8.07999992   1.456000e+01]
// [5.6 7.8]           [2.0 1.0]   [2.104000e+01 2.752000e+01]

// CHECK: output #0: dense<9.680000e+00> : tensor<1x1x1x1xf32>

// TINY_SQUARE: output #0: dense<[
// TINY_SQUARE-SAME:     [8.07999992, 1.456000e+01], [2.104000e+01, 2.752000e+01]
// TINY_SQUARE-SAME:   ]> : tensor<1x1x2x2xf32>

// Check with zeros as well. The result should be identically zeros.
// If any uninitialized data sneaks in (even very small values that would be
// rounding errors for the test case above), it will show up here.
// ZEROS: output #0: dense<0.000000e+00> : tensor<1x1x1x1xf32>

func @conv_2d_nchw_bias(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?x?x?x?xf32>, %arg2: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = tcf.conv_2d_nchw_bias %arg0, %arg1, %arg2 : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
