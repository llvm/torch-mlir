// RUN: npcomp-opt -lower-shaped-results-to-memref <%s -split-input-file | FileCheck %s --dump-input=fail

#map0 = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func @linalg_generic
func @linalg_generic(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xindex>) -> tensor<?xf32> {
  // CHECK: %[[LHS:.*]] = tcp.tensor_to_memref %arg0 : tensor<?xf32> -> memref<?xf32>
  // CHECK: %[[RHS:.*]] = tcp.tensor_to_memref %arg1 : tensor<?xf32> -> memref<?xf32>
  // CHECK: %[[DST:.*]] = tcp.alloc_memref %arg2 : memref<?xf32>
  // CHECK: linalg.generic {{.*}} %[[LHS]], %[[RHS]], %[[DST]]
  // CHECK-NOT: tcp.shaped_results
  %0 = tcp.shaped_results %arg2 {
    %0 = linalg.generic {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} %arg0, %arg1 {
    ^bb0(%arg3: f32, %arg4: f32):
      %8 = addf %arg3, %arg4 : f32
      linalg.yield %8 : f32
    } : tensor<?xf32>, tensor<?xf32> -> tensor<?xf32>
    tcp.yield %0 : tensor<?xf32>
  } : tensor<?xindex> -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @tcp_broadcast_to
func @tcp_broadcast_to(%arg0: tensor<?xf32>, %arg1: tensor<?xindex>) -> tensor<?x?xf32> {
  // Check for two nested loops, but don't look at more detail for now.
  // TODO: This pass should not create loops. Instead it should create a
  // buffer version of tcp.broadcast_to
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK-NOT: tcp.shaped_results
  %0 = tcp.shaped_results %arg1 {
    %0 = "tcp.broadcast_to"(%arg0, %arg1) : (tensor<?xf32>, tensor<?xindex>) -> tensor<?x?xf32>
    tcp.yield %0 : tensor<?x?xf32>
  } : tensor<?xindex> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL:   func @tcp_matmul(
// CHECK-SAME:                     %arg0: tensor<?x?xf32>,
// CHECK-SAME:                     %arg1: tensor<?x?xf32>,
// CHECK-SAME:                     %[[SHAPE:.*]]: tensor<?xindex>) -> tensor<?x?xf32> {
// CHECK:           %[[LHS:.*]] = tcp.tensor_to_memref %arg0 : tensor<?x?xf32> -> memref<?x?xf32>
// CHECK:           %[[RHS:.*]] = tcp.tensor_to_memref %arg1 : tensor<?x?xf32> -> memref<?x?xf32>
// CHECK:           %[[RESULT:.*]] = tcp.alloc_memref %[[SHAPE]] : memref<?x?xf32>
// CHECK:           linalg.matmul ins(%[[LHS]], %[[RHS]] : memref<?x?xf32>, memref<?x?xf32>) outs(%[[RESULT]] : memref<?x?xf32>)
// CHECK:           %[[RET:.*]] = tcp.memref_to_tensor %[[RESULT]] : memref<?x?xf32> -> tensor<?x?xf32>
// CHECK:           return %[[RET]] : tensor<?x?xf32>
// CHECK:         }
func @tcp_matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %shape: tensor<?xindex>) -> tensor<?x?xf32> {
  %0 = tcp.shaped_results %shape {
    %matmul = tcp.matmul %arg0, %arg1 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    tcp.yield %matmul : tensor<?x?xf32>
  } : tensor<?xindex> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
