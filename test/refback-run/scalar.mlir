// RUN: refback-run %s \
// RUN:   -invoke scalar_arg \
// RUN:   -arg-value="2.5 : f32" \
// RUN:   -shared-libs=%npcomp_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s

// CHECK: output #0: 2.500000e+00 : f32
func @scalar_arg(%arg0: f32) -> f32 {
  return %arg0 : f32
}
