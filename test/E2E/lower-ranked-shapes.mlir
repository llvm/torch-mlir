// RUN: npcomp-opt -lower-ranked-shapes <%s | FileCheck %s --dump-input=fail


// CHECK-LABEL: func @broadcast_rank2_rank1
func @broadcast_rank2_rank1(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>) -> (index, index) {
  // CHECK-NOT: shape.shape_of
  // CHECK-NOT: shape.broadcast
  // CHECK-NOT: tcp.get_extent
  %0 = shape.shape_of %arg0 : tensor<?x?xf32>
  %1 = shape.shape_of %arg1 : tensor<?xf32>
  %2 = "shape.broadcast"(%0, %1) : (!shape.shape, !shape.shape) -> !shape.shape
  %e0 = tcp.get_extent %2, 0
  %e1 = tcp.get_extent %2, 1
  return %e0, %e1 : index, index
}
