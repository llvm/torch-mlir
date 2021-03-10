// RUN: npcomp-opt --convert-aten-to-tcf %s | npcomp-run-mlir \
// RUN:   -invoke aten_conv_2d_nchw \
// RUN:   -arg-value="dense<0.0> : tensor<2x1x1x1xf32>" \
// RUN:   -arg-value="dense<0.0> : tensor<1x1x1x1xf32>" \
// RUN:   -arg-value="dense<0.0> : tensor<1x1x1x1xf32>" \
// RUN:   -shared-libs=%npcomp_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BATCH

// RUN: npcomp-opt --convert-aten-to-tcf %s | npcomp-run-mlir \
// RUN:   -invoke aten_conv_2d_nchw \
// RUN:   -arg-value="dense<0.0> : tensor<1x2x1x1xf32>" \
// RUN:   -arg-value="dense<0.0> : tensor<2x2x1x1xf32>" \
// RUN:   -arg-value="dense<0.0> : tensor<1x1x1x1xf32>" \
// RUN:   -shared-libs=%npcomp_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SAME_CHANNELS

// RUN: npcomp-opt --convert-aten-to-tcf %s | npcomp-run-mlir \
// RUN:   -invoke aten_conv_2d_nchw \
// RUN:   -arg-value="dense<0.0> : tensor<1x2x1x1xf32>" \
// RUN:   -arg-value="dense<0.0> : tensor<1x2x1x1xf32>" \
// RUN:   -arg-value="dense<0.0> : tensor<1x1x1x1xf32>" \
// RUN:   -shared-libs=%npcomp_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DIFFERENT_CHANNELS

// RUN: npcomp-opt --convert-aten-to-tcf %s | npcomp-run-mlir \
// RUN:   -invoke aten_conv_2d_nchw \
// RUN:   -arg-value="dense<0.0> : tensor<1x1x2x2xf32>" \
// RUN:   -arg-value="dense<0.0> : tensor<1x1x1x1xf32>" \
// RUN:   -arg-value="dense<0.0> : tensor<1x1x1x1xf32>" \
// RUN:   -shared-libs=%npcomp_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s --check-prefix=TINY_SQUARE

// RUN: npcomp-opt --convert-aten-to-tcf %s | npcomp-run-mlir \
// RUN:   -invoke aten_conv_2d_nchw \
// RUN:   -arg-value="dense<0.0> : tensor<1x1x32x32xf32>" \
// RUN:   -arg-value="dense<0.0> : tensor<1x1x32x32xf32>" \
// RUN:   -arg-value="dense<0.0> : tensor<1x1x1x1xf32>" \
// RUN:   -shared-libs=%npcomp_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s --check-prefix=HUGE_SQUARE

// RUN: npcomp-opt --convert-aten-to-tcf %s | npcomp-run-mlir \
// RUN:   -invoke aten_conv_2d_nchw \
// RUN:   -arg-value="dense<0.0> : tensor<1x1x2x2xf32>" \
// RUN:   -arg-value="dense<0.0> : tensor<1x1x0x0xf32>" \
// RUN:   -arg-value="dense<0.0> : tensor<1x1x1x1xf32>" \
// RUN:   -shared-libs=%npcomp_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ZERO_KH_KW

// RUN: npcomp-opt --convert-aten-to-tcf %s | npcomp-run-mlir \
// RUN:   -invoke aten_conv_2d_nchw \
// RUN:   -arg-value="dense<0.0> : tensor<1x1x0x0xf32>" \
// RUN:   -arg-value="dense<0.0> : tensor<1x1x0x0xf32>" \
// RUN:   -arg-value="dense<0.0> : tensor<1x1x1x1xf32>" \
// RUN:   -shared-libs=%npcomp_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ZERO_H_W

// BATCH: output #0: dense<0.000000e+00> : tensor<2x1x1x1xf32>

// SAME_CHANNELS: output #0: dense<0.000000e+00> : tensor<1x2x1x1xf32>

// DIFFERENT_CHANNELS: output #0: dense<0.000000e+00> : tensor<1x1x1x1xf32>

// TINY_SQUARE: output #0: dense<0.000000e+00> : tensor<1x1x2x2xf32>

// HUGE_SQUARE: output #0: dense<0.000000e+00> : tensor<1x1x1x1xf32>

// ZERO_KH_KW: output #0: dense<0.000000e+00> : tensor<1x1x3x3xf32>

// ZERO_H_W: output #0: dense<0.000000e+00> : tensor<1x1x1x1xf32>

func @aten_conv_2d_nchw(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?x?x?x?xf32>, %arg2: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %c0_i64 = constant 0 : i64
  %c1_i64 = constant 1 : i64
  %0 = basicpy.build_list %c1_i64, %c1_i64 : (i64, i64) -> !basicpy.ListType
  %1 = basicpy.build_list %c0_i64, %c0_i64 : (i64, i64) -> !basicpy.ListType
  %2 = basicpy.build_list %c1_i64, %c1_i64 : (i64, i64) -> !basicpy.ListType
  %6 = "aten.conv2d"(%arg0, %arg1, %arg2, %0, %1, %2, %c1_i64) : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, !basicpy.ListType, !basicpy.ListType, !basicpy.ListType, i64) -> tensor<?x?x?x?xf32>
  return %6 : tensor<?x?x?x?xf32>
}
