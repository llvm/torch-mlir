// RUN: npcomp-opt -lower-to-hybrid-tensor-memref-pipeline <%s | FileCheck %s --dump-input=fail

#map0 = affine_map<(d0) -> (d0)>
func @f(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "shape.shape_of"(%arg0) : (tensor<?xf32>) -> !shape.shape
  %1 = "shape.shape_of"(%arg1) : (tensor<?xf32>) -> !shape.shape
  %2 = "shape.broadcast"(%0, %1) : (!shape.shape, !shape.shape) -> !shape.shape
  %3 = "shape.abort_if_error"(%2) : (!shape.shape) -> none
  %4 = "tcp.island"(%3) ( {
    %5 = "tcp.broadcast_to"(%arg0, %2) : (tensor<?xf32>, !shape.shape) -> tensor<?xf32>
    %6 = "tcp.broadcast_to"(%arg1, %2) : (tensor<?xf32>, !shape.shape) -> tensor<?xf32>
    // CHECK: tcp.alloc_memref
    %7 = linalg.generic {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} %5, %6 {
    ^bb0(%arg2: f32, %arg3: f32):     // no predecessors
      %8 = addf %arg2, %arg3 : f32
      linalg.yield %8 : f32
    }: tensor<?xf32>, tensor<?xf32> -> tensor<?xf32>
    "tcp.yield"(%7) : (tensor<?xf32>) -> ()
  }) : (none) -> tensor<?xf32>
  return %4 : tensor<?xf32>
}

