// RUN: not npcomp-run-mlir %s \
// RUN:   -invoke expects_one_tensor \
// RUN:   -arg-value="1.0 : f32" \
// RUN:   -shared-libs=%npcomp_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s

// CHECK: invoking 'expects_one_tensor': input argument type mismatch.
// CHECK-SAME: actual (provided by user): Float
// CHECK-SAME: expected (from compiler): kTensor
func @expects_one_tensor(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = tcf.add %arg0, %arg0 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}