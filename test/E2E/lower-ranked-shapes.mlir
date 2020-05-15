// RUN: npcomp-opt -lower-ranked-shapes <%s | FileCheck %s --dump-input=fail


// CHECK-LABEL: func @broadcast_rank2_rank1
func @broadcast_rank2_rank1(%arg0: index, %arg1: index, %arg2: index) -> (index, index) {
  // CHECK-NOT: shape.broadcast
  // CHECK-NOT: tcp.get_extent
  %0 = "tcp.shape_from_extents"(%arg0, %arg1) : (index, index) -> !shape.shape
  %1 = "tcp.shape_from_extents"(%arg2) : (index) -> !shape.shape
  %2 = "shape.broadcast"(%0, %1) : (!shape.shape, !shape.shape) -> !shape.shape
  %e0 = tcp.get_extent %2, 0
  %e1 = tcp.get_extent %2, 1
  return %e0, %e1 : index, index
}
