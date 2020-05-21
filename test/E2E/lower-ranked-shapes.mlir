// RUN: npcomp-opt -lower-ranked-shapes <%s | FileCheck %s --dump-input=fail


// CHECK-LABEL: func @broadcast_rank2_rank1
func @broadcast_rank2_rank1(%arg0: index, %arg1: index, %arg2: index) -> (index, index) {
  // CHECK-NOT: shape.broadcast
  // CHECK-NOT: tcp.get_extent
  %0 = shape.from_extents %arg0, %arg1
  %1 = shape.from_extents %arg2
  %2 = "shape.broadcast"(%0, %1) : (!shape.shape, !shape.shape) -> !shape.shape
  %e0 = tcp.get_extent %2, 0
  %e1 = tcp.get_extent %2, 1
  return %e0, %e1 : index, index
}

// CHECK-LABEL: func @erase_shape_observe_error
func @erase_shape_observe_error(%arg0: index) {
  // CHECK-NOT tcp.shape_observe_error
  %0 = shape.from_extents %arg0
  "tcp.shape_observe_error"(%0) : (!shape.shape) -> none
  return
}
