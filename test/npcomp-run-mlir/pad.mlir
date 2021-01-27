// RUN: npcomp-run-mlir %s \
// RUN:   -invoke pad \
// RUN:   -arg-value="dense<[1.2, 3.4]> : tensor<2xf32>" \
// RUN:   -shared-libs=%npcomp_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s

// CHECK: output #0: dense<
// CHECK-SAME:   [0.000000e+00, 1.200000e+00, 3.400000e+00, 0.000000e+00, 0.000000e+00]
// CHECK-SAME: > : tensor<5xf32>

func @pad(%arg0: tensor<?xf32> ) -> tensor<?xf32> {
  %lowerExpansion = shape.const_shape [1] : tensor<?xindex>
  %upperExpansion = shape.const_shape [2] : tensor<?xindex>
  %fillVal = constant 0.0 : f32
  %0 = tcp.pad %arg0, %lowerExpansion, %upperExpansion, %fillVal : (tensor<?xf32>, tensor<?xindex>, tensor<?xindex>, f32) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

