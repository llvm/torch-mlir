// RUN: npcomp-opt --convert-aten-to-tcf %s | not npcomp-run-mlir \
// RUN:   -invoke aten_conv_2d_nchw \
// RUN:   -arg-value="dense<0.0> : tensor<1x1x2x2xf32>" \
// RUN:   -arg-value="dense<0.0> : tensor<1x2x2x2xf32>" \
// RUN:   -arg-value="dense<0.0> : tensor<1x1x1x1xf32>" \
// RUN:   -shared-libs=%npcomp_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHANNELS

// RUN: npcomp-opt --convert-aten-to-tcf %s | not npcomp-run-mlir \
// RUN:   -invoke aten_conv_2d_nchw \
// RUN:   -arg-value="dense<0.0> : tensor<1x1x2x2xf32>" \
// RUN:   -arg-value="dense<0.0> : tensor<1x1x3x2xf32>" \
// RUN:   -arg-value="dense<0.0> : tensor<1x1x1x1xf32>" \
// RUN:   -shared-libs=%npcomp_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s --check-prefix=HEIGHT

// RUN: npcomp-opt --convert-aten-to-tcf %s | not npcomp-run-mlir \
// RUN:   -invoke aten_conv_2d_nchw \
// RUN:   -arg-value="dense<0.0> : tensor<1x1x2x2xf32>" \
// RUN:   -arg-value="dense<0.0> : tensor<1x1x2x3xf32>" \
// RUN:   -arg-value="dense<0.0> : tensor<1x1x1x1xf32>" \
// RUN:   -shared-libs=%npcomp_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s --check-prefix=WIDTH

// CHANNELS: NPCOMP: aborting: input and filter in-channels must be equal
// HEIGHT: NPCOMP: aborting: input height must be greater than or equal to filter KH-dimension
// WIDTH: NPCOMP: aborting: input width must be greater than or equal to filter KW-dimension
func @aten_conv_2d_nchw(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?x?x?x?xf32>, %arg2: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %c0_i64 = constant 0 : i64
  %c1_i64 = constant 1 : i64
  %0 = basicpy.build_list %c1_i64, %c1_i64 : (i64, i64) -> !basicpy.ListType
  %1 = basicpy.build_list %c0_i64, %c0_i64 : (i64, i64) -> !basicpy.ListType
  %2 = basicpy.build_list %c1_i64, %c1_i64 : (i64, i64) -> !basicpy.ListType
  %3 = "aten.conv2d"(%arg0, %arg1, %arg2, %0, %1, %2, %c1_i64) : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, !basicpy.ListType, !basicpy.ListType, !basicpy.ListType, i64) -> tensor<?x?x?x?xf32>
  return %3 : tensor<?x?x?x?xf32>
}
