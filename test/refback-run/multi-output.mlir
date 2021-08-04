// RUN: refback-run %s \
// RUN:   -invoke multi_output \
// RUN:   -arg-value="dense<1.0> : tensor<1xf32>" \
// RUN:   -shared-libs=%npcomp_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s

// CHECK: output #0: dense<2.000000e+00> : tensor<1xf32>
// CHECK: output #1: dense<2.000000e+00> : tensor<1xf32>
#map0 = affine_map<(d0) -> (d0)>
func @multi_output(%arg0: tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>) {
  %c0 = constant 0 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?xf32>
  %1 = linalg.init_tensor [%0] : tensor<?xf32>
  %2 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg0, %arg0 : tensor<?xf32>, tensor<?xf32>) outs(%1 : tensor<?xf32>) {
  ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
    %6 = addf %arg2, %arg3 : f32
    linalg.yield %6 : f32
  } -> tensor<?xf32>
  return %2, %2 : tensor<?xf32>, tensor<?xf32>
}
