// RUN: npcomp-opt -bypass-shapes <%s | FileCheck %s --dump-input=fail

#map0 = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func @linalg_generic
func @linalg_generic(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  // This is an elementwise linalg op, so output shape is equal to input shape.
  // CHECK: %[[SHAPE:.*]] = shape.shape_of %arg0
  // CHECK: tcp.shaped_results %[[SHAPE]]
  %0 = linalg.generic {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} %arg0, %arg1 {
  ^bb0(%arg2: f32, %arg3: f32):
    %8 = addf %arg2, %arg3 : f32
    linalg.yield %8 : f32
  }: tensor<?xf32>, tensor<?xf32> -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @tcp_broadcast_to
func @tcp_broadcast_to(%arg0: tensor<?xf32>, %arg1: tensor<?xindex>) {
  // CHECK: %0 = tcp.shaped_results %arg1
  %0 = "tcp.broadcast_to"(%arg0, %arg1) : (tensor<?xf32>, tensor<?xindex>) -> tensor<?x?xf32>
  return
}
