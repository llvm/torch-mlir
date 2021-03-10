// RUN: not npcomp-run-mlir %s \
// RUN:   -invoke invalid_input_shape \
// RUN:   -arg-value="dense<1.0> : tensor<2x2x2x2xf32>" \
// RUN:   -shared-libs=%npcomp_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s -check-prefix=ARG0-INVALID

// RUN: not npcomp-run-mlir %s \
// RUN:   -invoke invalid_input_shape_arg1 \
// RUN:   -arg-value="dense<1.0> : tensor<1x2x5xf32>" \
// RUN:   -arg-value="dense<1.0> : tensor<1x2x10xf32>" \
// RUN:   -shared-libs=%npcomp_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s -check-prefix=ARG1-INVALID

// ARG0-INVALID: invoking 'invalid_input_shape': input shape mismatch (%arg0).
// ARG0-INVALID-SAME: actual (provided by user): (2x2x2x2)
// ARG0-INVALID-SAME: expected (from compiler): (1x2x3x4)
func @invalid_input_shape(%arg0: tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32> {
  return %arg0: tensor<1x2x3x4xf32>
}

// ARG1-INVALID: invoking 'invalid_input_shape_arg1': input shape mismatch (%arg1)
// ARG1-INVALID-SAME: actual (provided by user): (1x2x10)
// ARG1-INVALID-SAME: expected (from compiler): (1x4x?)
func @invalid_input_shape_arg1(%arg0: tensor<1x2x?xf32>, %arg1: tensor<1x4x?xf32>) {
  return 
}