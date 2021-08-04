// RUN: refback-run %s \
// RUN:   -invoke forward \
// RUN:   -arg-value="dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>" \
// RUN:   -arg-value="dense<[10.0, 20.0]> : tensor<2xf32>" \
// RUN:   -shared-libs=%npcomp_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s

// CHECK{LITERAL}: output #0: dense<[[1.100000e+01, 2.200000e+01], [1.300000e+01, 2.400000e+01]]> : tensor<2x2xf32>
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>

builtin.func @forward(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>) -> tensor<?x?xf32> {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = tensor.dim %arg1, %c0 : tensor<?xf32>
  %3 = cmpi eq, %1, %2 : index
  assert %3, "mismatched size for broadcast"
  %4 = linalg.init_tensor [%0, %1] : tensor<?x?xf32>
  %5 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?xf32>) outs(%4 : tensor<?x?xf32>) {
  ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
    %6 = addf %arg2, %arg3 : f32
    linalg.yield %6 : f32
  } -> tensor<?x?xf32>
  return %5 : tensor<?x?xf32>
}
