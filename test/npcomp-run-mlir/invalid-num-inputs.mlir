// RUN: not npcomp-run-mlir %s \
// RUN:   -invoke requires_one_input \
// RUN:   -shared-libs=%npcomp_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s

// CHECK: invoking 'requires_one_input': expected 1 inputs
func @requires_one_input(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  return %arg0 : tensor<?xf32>
}
