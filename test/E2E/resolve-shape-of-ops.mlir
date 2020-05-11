// RUN: npcomp-opt -resolve-shape-of-ops <%s -split-input-file -verify-diagnostics | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @basic
func @basic(%arg0: !shape.shape) -> !shape.shape {
  %memref = tcp.alloc_memref %arg0 : memref<?xf32>
  %tensor = tensor_load %memref : memref<?xf32>
  %shape = "shape.shape_of"(%tensor) : (tensor<?xf32>) -> !shape.shape
  // CHECK: return %arg0
  return %shape : !shape.shape
}

// -----

// CHECK-LABEL: func @arg_unresolved_ok
func @arg_unresolved_ok(%arg0: tensor<?xf32>) -> !shape.shape {
  %0 = "shape.shape_of"(%arg0): (tensor<?xf32>) -> !shape.shape
  return %0 : !shape.shape
}

// -----

// CHECK-LABEL: func @TODO_bb_arg_unresolved_not_ok
// TODO: This should emit a diagnostic, but doesn't. Why?
// addDynamicallyLegalOp isn't working as I expect.
func @TODO_bb_arg_unresolved_not_ok(%arg0: i1, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> !shape.shape {
  cond_br %arg0, ^bb1(%arg1: tensor<?xf32>), ^bb1(%arg2: tensor<?xf32>)
^bb1(%bbarg: tensor<?xf32>):
  %0 = "shape.shape_of"(%bbarg): (tensor<?xf32>) -> !shape.shape
  return %0 : !shape.shape
}
