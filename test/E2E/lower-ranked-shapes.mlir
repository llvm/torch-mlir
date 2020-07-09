// RUN: npcomp-opt -lower-ranked-shapes <%s -split-input-file -verify-diagnostics | FileCheck %s --dump-input=fail


// CHECK-LABEL: func @broadcast_rank2_rank1
func @broadcast_rank2_rank1(%arg0: index, %arg1: index, %arg2: index) -> (index, index) {
  // CHECK-NOT: shape.broadcast
  // CHECK-NOT: tcp.get_extent
  // CHECK-NOT: shape.from_extents
  %0 = shape.from_extents %arg0, %arg1
  %1 = shape.from_extents %arg2
  %2 = "shape.broadcast"(%0, %1) : (!shape.shape, !shape.shape) -> !shape.shape
  %e0 = tcp.get_extent %2, 0
  %e1 = tcp.get_extent %2, 1
  return %e0, %e1 : index, index
}

// CHECK-LABEL: func @erase_stray_shape_ops
func @erase_stray_shape_ops(%arg0: index) {
  // CHECK-NOT: tcp.shape_observe_error
  // CHECK-NOT: shape.from_extents
  %0 = shape.from_extents %arg0
  "tcp.shape_observe_error"(%0) : (!shape.shape) -> none
  return
}

// -----

func @cannot_erase_stray_shape_ops() -> !shape.shape {
  // expected-error @+1 {{could not be eliminated}}
  %0 = shape.from_extents
  return %0 : !shape.shape
}

// -----

// CHECK-LABEL: func @const_shape
func @const_shape() -> index {
  // CHECK-NOT: shape.const_shape
  %0 = shape.const_shape []
  %1 = shape.const_shape [7]
  %2 = tcp.get_extent %1, 0
  // CHECK: %[[C7:.*]] = constant 7 : index
  // CHECK: return %[[C7]]
  return %2 : index
}
