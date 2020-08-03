// RUN: npcomp-opt -lower-ranked-shapes <%s -split-input-file -verify-diagnostics | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @broadcast_rank2_rank1
func @broadcast_rank2_rank1(%arg0: index, %arg1: index, %arg2: index) -> (index, index) {
  // CHECK-NOT: shape.broadcast
  // CHECK-NOT: tcp.get_extent
  // CHECK-NOT: shape.from_extents
  %0 = shape.from_extents %arg0, %arg1
  %1 = shape.to_extent_tensor %0 : !shape.shape -> tensor<?xindex>
  %2 = shape.from_extents %arg2
  %3 = shape.to_extent_tensor %2 : !shape.shape -> tensor<?xindex>
  %4 = "shape.broadcast"(%1, %3) : (tensor<?xindex>, tensor<?xindex>) -> !shape.shape
  %5 = shape.to_extent_tensor %4 : !shape.shape -> tensor<?xindex>
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %e0 = shape.get_extent %5, %c0 : tensor<?xindex>, index -> index
  %e1 = shape.get_extent %5, %c1 : tensor<?xindex>, index -> index
  return %e0, %e1 : index, index
}

// -----
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
// TODO: Remove this as it is now just testing shape and std ops.
// CHECK-LABEL: func @const_shape
func @const_shape() -> index {
  // CHECK-NOT: shape.const_shape
  %0 = shape.const_shape [] : tensor<?xindex>
  %1 = shape.const_shape [7] : tensor<?xindex>
  %2 = constant 0 : index
  %3 = shape.get_extent %1, %2 : tensor<?xindex>, index -> index
  // CHECK: %[[C7:.*]] = constant 7 : index
  // CHECK: return %[[C7]]
  return %3 : index
}
