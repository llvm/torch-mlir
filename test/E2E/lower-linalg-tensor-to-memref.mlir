// RUN: npcomp-opt -lower-linalg-tensor-to-memref <%s | FileCheck %s --dump-input=fail
#map0 = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func @f
func @f(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK-DAG: %[[LHS:.+]] = "tcp.alloc_memref"
  // CHECK-DAG: %[[RHS:.+]] = "tcp.alloc_memref"
  // CHECK-DAG: %[[DST:.+]] = "tcp.alloc_memref"
  // CHECK: linalg.generic{{.*}} %[[LHS]], %[[RHS]], %[[DST]]
  %0 = linalg.generic {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} %arg0, %arg1 {
  ^bb0(%arg2: f32, %arg3: f32):
    %8 = addf %arg2, %arg3 : f32
    linalg.yield %8 : f32
  }: tensor<?xf32>, tensor<?xf32> -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
