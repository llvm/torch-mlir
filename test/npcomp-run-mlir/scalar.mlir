// RUN: npcomp-run-mlir %s \
// RUN:   -invoke scalar \
// RUN:   -arg-value="dense<1.0> : tensor<f32>" \
// RUN:   -shared-libs=%npcomp_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SCALAR

// RUN: npcomp-run-mlir %s \
// RUN:   -invoke scalar_arg \
// RUN:   -arg-value="2.5 : f32" \
// RUN:   -shared-libs=%npcomp_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SCALAR_ARG

// SCALAR: output #0: dense<2.000000e+00> : tensor<f32>
func @scalar(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = tcf.add %arg0, %arg0 : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// SCALAR_ARG: output #0: 2.500000e+00 : f32
func @scalar_arg(%arg0: f32) -> f32 {
  return %arg0 : f32
}