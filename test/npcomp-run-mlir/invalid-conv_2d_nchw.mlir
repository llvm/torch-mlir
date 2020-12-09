// RUN: not npcomp-run-mlir %s \
// RUN:   -invoke conv_2d_nchw \
// RUN:   -arg-value="dense<[[[[1.0, 0.0], [1.0, 1.0]]]]> : tensor<1x1x2x2xf32>" \
// RUN:   -arg-value="dense<[[[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]]]> : tensor<1x2x2x2xf32>" \
// RUN:   -shared-libs=%npcomp_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHANNELS

// RUN: not npcomp-run-mlir %s \
// RUN:   -invoke conv_2d_nchw \
// RUN:   -arg-value="dense<[[[[1.0, 0.0], [1.0, 1.0]]]]> : tensor<1x1x2x2xf32>" \
// RUN:   -arg-value="dense<[[[[1.0, 2.0], [3.0, 4.0]]], [[[1.0, 2.0], [3.0, 4.0]]]]> : tensor<2x1x2x2xf32>" \
// RUN:   -shared-libs=%npcomp_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHANNELSF

// RUN: not npcomp-run-mlir %s \
// RUN:   -invoke conv_2d_nchw \
// RUN:   -arg-value="dense<[[[[1.0, 0.0], [1.0, 1.0]]]]> : tensor<1x1x2x2xf32>" \
// RUN:   -arg-value="dense<[[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]]> : tensor<1x1x3x2xf32>" \
// RUN:   -shared-libs=%npcomp_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s --check-prefix=HEIGHT

// RUN: not npcomp-run-mlir %s \
// RUN:   -invoke conv_2d_nchw \
// RUN:   -arg-value="dense<[[[[1.0, 0.0], [1.0, 1.0]]]]> : tensor<1x1x2x2xf32>" \
// RUN:   -arg-value="dense<[[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]]> : tensor<1x1x2x3xf32>" \
// RUN:   -shared-libs=%npcomp_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s --check-prefix=WIDTH

// CHANNELS: NPCOMP: aborting: input and filter channels must be equal
// CHANNELSF: NPCOMP: aborting: input channels and filter F-dimension must be equal
// HEIGHT: NPCOMP: aborting: input height must be greater than or equal to filter KH-dimension
// WIDTH: NPCOMP: aborting: input width must be greater than or equal to filter KW-dimension
func @conv_2d_nchw(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = tcf.conv_2d_nchw %arg0, %arg1 : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
